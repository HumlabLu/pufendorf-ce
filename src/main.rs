use iced::widget::text::LineHeight;
use iced::widget::operation::snap_to;
use iced::widget::scrollable::RelativeOffset;
use iced::widget::Id;
use log::{debug, error, info, trace, LevelFilter};
use flexi_logger::{DeferredNow, Record};
use flexi_logger::{Duplicate, FileSpec, LogSpecification, Logger, WriteMode};
use iced::{
    widget::{button, column, container, row, scrollable, slider, text, text_input},
    Element, Length, Settings, Task, Theme,
};
use std::{
    sync::{Arc, LazyLock, Mutex},
};
use async_stream::stream;
use tokio_stream::StreamExt;
use serde_json::Value;
use std::fs;
use std::path::Path;
use openai_dive::v1::api::Client;
use openai_dive::v1::endpoints::chat::RoleTrackingStream;
use openai_dive::v1::resources::chat::{
    ChatCompletionParametersBuilder, ChatCompletionResponseFormat, ChatMessage, ChatMessageContent,
    DeltaChatMessage,
};
use openai_dive::v1::models::ModerationModel;
use openai_dive::v1::resources::moderation::{ModerationInput, ModerationParametersBuilder};
use clap::Parser;
use std::io::Write;
use std::str::FromStr;

mod lance;
use lance::{create_database, create_empty_table, append_documents, get_row_count, dump_table};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use arrow_array::{
    RecordBatch, UInt64Array
};
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::index::scalar::FullTextSearchQuery;use iced::futures::TryStreamExt;
use lancedb::rerankers::rrf::RRFReranker;
use arrow_array::{Float32Array, StringArray, Array};
use tokio::runtime::Runtime;

mod structs;
use structs::*;

mod embedder;
use embedder::parse_embedding_model;

use std::env;
use std::collections::HashMap;
use fastembed::{TextRerank, RerankInitOptions, RerankerModel};

// LOG is the Id for the chat log output pane, needed in the snap_to(...) function.
static LOG: LazyLock<Id> = LazyLock::new(|| Id::new("log"));

#[derive(Parser)]
#[command(version, about, long_about = "Reading data.")]
struct Cli {
    /// Sets the level of logging;
    /// error, warn, info, debug, or trace
    #[arg(short, long, default_value = "info")]
    log_level: String,

    #[arg(short, long, help = "Append text file with info.")]
    append: Option<String>,

    #[arg(long, help = "Chunk size.", default_value_t = 4096)]
    chunksize: usize,

    #[arg(short, long, help = "Retrieval cut off.", default_value_t = 1.5)]
    cutoff: f32,

    #[arg(short, long, help = "DB name.")]
    dbname: Option<String>,

    #[arg(short, long, help = "Embedding model.", default_value = "all-minilm-l6-v2")]
    embedmodel: String,

    #[arg(long, help = "Dump DB contents.", default_value_t = 0)]
    dump: usize,

    #[arg(short, long, help = "Text file with info.")]
    filename: Option<String>,

    #[arg(long, help = "Font size.", default_value_t = 18)]
    fontsize: u32,

    #[arg(short, long, help = "System prompt/info json file.")]
    promptfile: Option<String>,

    #[arg(short, long, help = "Table name.")]
    tablename: Option<String>,
}

fn log_format(
    w: &mut dyn Write,
    now: &mut DeferredNow,
    record: &Record,
) -> Result<(), std::io::Error> {
    let file_path = record.file().unwrap_or("<unknown>");
    let file_name = Path::new(file_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("<unknown>");
    let line = record.line().unwrap_or(0);
    write!(
        w,
        "{} [{}] {}:{} - {}",
        now.format("%Y-%m-%d %H:%M:%S"),
        record.level(),
        file_name,
        line,
        &record.args()
    )
}


fn theme(_: &App) -> Theme {
    Theme::TokyoNight //GruvboxLight //Dark
    // Theme::GruvboxDark // Light
}


async fn connect_db(db_name: String) -> lancedb::Result<lancedb::Connection> {
    lancedb::connect(&db_name).execute().await
}

/*
    We have too many structs -- a struct for the Iced app, and another AppConfig
    struct to move settings from main to the app.

    The AppConfig should prolly also contain the embedding model and DB connexion.
*/
fn main() -> iced::Result {
    let cli = Cli::parse();

    // This switches off logging from html5 and other crates.
    let level_filter = LevelFilter::from_str(&cli.log_level).unwrap_or(LevelFilter::Off);
    let log_spec = LogSpecification::builder()
        .module("html5ever", LevelFilter::Off)
        .module("rusty_puff", level_filter) // Sets our level to the one on the cli.
        .build();

    let _logger = Logger::with(log_spec)
        .format(log_format)
        .log_to_file(
            FileSpec::default()
                .suppress_timestamp()
                .basename("pufenbot")
                .suffix("log"),
        )
        .append()
        .duplicate_to_stderr(Duplicate::Info) // was ::All
        .write_mode(WriteMode::BufferAndFlush)
        .start().expect("Logging?");
    info!("Start");

    let _oaik = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY");

    // ------------ New Db stuff
    
    let embedmodel = parse_embedding_model(&cli.embedmodel).expect("Error");

    // Embedding model (downloads once, then runs locally).
    // See also embedder.rs
    let mut embedder = TextEmbedding::try_new(
        InitOptions::new(embedmodel.clone()).with_show_download_progress(true),
    ).expect("No embedding model.");

    // Embedder, plus determine dimension.
    let model_info = TextEmbedding::get_model_info(&embedmodel);
    let dim = model_info.unwrap().dim as i32;
    info!("Embedding dim {}", dim);

    let _one_embeddings = embedder.embed(&["one"], None).expect("Cannot embed?");

    // My DB. Created if it doesn't exist.
    let db_name = match cli.dbname {
        Some(dbn) => dbn,
        None => "data/lancedb_fastembed".to_string()
    };
    info!("Database: {db_name}");

    let table_name = match cli.tablename {
        Some(tbln) => tbln,
        None => "docs".to_string()
    };
    info!("Table name: {table_name}");

    if let Some(ref filename) = cli.filename {
        info!("Filename {filename}.");
        let rt = Runtime::new().unwrap();
        // cli.chunksize
        let _ = rt.block_on(create_database(filename, &db_name, &table_name, &mut embedder, cli.chunksize));
    }

    let promptfile = match cli.promptfile {
        Some(pf) => pf,
        None => "assets/chatprompts.json".to_string()
    };

    if let Some(ref filename) = cli.append{
        info!("Append filename {filename}.");
        let rt = Runtime::new().unwrap();
        let _ = rt.block_on(append_documents(filename, &db_name, &table_name, cli.chunksize));
    }

    let rt = Runtime::new().unwrap();
    let _db: Option<lancedb::Connection> = match rt.block_on(connect_db(db_name.clone())) {
        Ok(db) => Some(db),
        Err(_e) => {
            error!("DB Error!");
            None
        }
    };

    // Create empty table if not exist?
    let _ = rt.block_on(create_empty_table(&db_name, &table_name));
    info!("Row count: {}", rt.block_on(get_row_count(&db_name, &table_name)));
    
    if cli.dump > 0 {
        let rt = Runtime::new().unwrap();
        let _ = rt.block_on(dump_table(&db_name, &table_name, cli.dump));
    }

    // Alternative way? which is best?
    let rt = Runtime::new().unwrap();
    let dbc: Option<lancedb::Connection> = match rt.block_on(connect_db(db_name.clone())) {
        Ok(db) => Some(db),
        Err(e) => {
            error!("DB Error! {e}");
            None
        }
    };

    // Have DB connexion here?
    let config = AppConfig {
        db_path: db_name.clone(),
        table_name: table_name.clone(),
        promptfile: promptfile,
        model: "gpt-4o-mini".into(),
        fontsize: cli.fontsize,
        cut_off: cli.cutoff,
        max_context: 12,
        db_connexion: Arc::new(Mutex::new(dbc)),
        embedder: Arc::new(Mutex::new(embedder)),
        chunk_size: cli.chunksize,
    };

    iced::application(
        move || App::new(config.clone()),
        App::update,
        App::view,
        )
        // iced::application(App::new, App::update, App::view)
        .title("Speak with Pufendorf")
        .theme(theme)
        .settings(Settings::default())
        .font(include_bytes!("../assets/FiraMono-Medium.ttf").as_slice())
        .run()
}

impl App {
    fn new(config: AppConfig) -> (Self, Task<Message>) {
        // Read the prompts from a json file.
        // Should contain a system_prompt and extra_info.
        let file_path = Path::new(&config.promptfile);
        let content = fs::read_to_string(file_path).expect("no file");
        let data: Value = serde_json::from_str(&content).expect("data");
        let sysprompt = data["system_prompt"]
            .as_str()
            .unwrap_or("Answer the questions.");
        let mut sysprompt = sysprompt.to_string();
        let label = data["label"]
            .as_str()
            .unwrap_or("Answer: ");
        if let Some(extras) = data["extra_info"].as_array() {
            sysprompt += "\n";
            for extra in extras {
                sysprompt += "\n";
                sysprompt += extra.as_str().unwrap_or("");
            }
        }
        debug!("{}", sysprompt);

        let history = Arc::new(Mutex::new(vec![
            Line {
                role: Role::System,
                content: sysprompt.clone(),
            }]
        ));

        (Self {
            config, 

            model: "gpt-4.1-nano".into(),
            mode: Mode::Chat,

            temperature: 0.1,
            num_predict: 2024,

            // Draft is user input, lines are the text in the output pane.
            draft: String::new(),
            lines: vec![Line {
                role: Role::System,
                content: "".into(),
            }],
            waiting: false,

            history,

            system_prompt: sysprompt,
            extra_info: "The year is 1667".into(), // Not used.
            label: label.to_string(),
        }, Task::none())
    }

    fn update(&mut self, msg: Message) -> Task<Message> {
        match msg {
            Message::DraftChanged(s) => {
                self.draft = s;
                Task::none()
            }

            Message::ModeChanged(m) => {
                self.mode = m;
                Task::none()
            }

            Message::TemperatureChanged(t) => {
                self.temperature = t;
                Task::none()
            }

            Message::CutOffChanged(t) => {
                self.config.cut_off = t;
                Task::none()
            }

            Message::MaxContextChanged(t) => {
                self.config.max_context = t;
                Task::none()
            }

            Message::NumPredictChanged(n) => {
                self.num_predict = n;
                Task::none()
            }

            Message::ResetParams => {
                self.temperature = 0.1;
                self.num_predict = 1024;
                Task::none()
            }

            Message::ClearAll => {
                self.draft.clear();
                self.waiting = false;
                self.lines = vec![Line {
                    role: Role::System,
                    content: "".into(),
                }];
                *self.history.lock().unwrap() = vec![Line {
                    role: Role::System,
                    content: self.system_prompt.clone(),
                    //"You are a helpful assistant.".to_string(),
                }];
                Task::none()
            }

            Message::Submit => {
                if self.waiting {
                    return Task::none();
                }
                let prompt = self.draft.trim().to_string();
                if prompt.is_empty() {
                    return Task::none();
                }

                self.draft.clear();
                self.waiting = true;

                self.lines.push(Line {
                    role: Role::User,
                    content: prompt.clone(),
                });
                self.lines.push(Line {
                    role: Role::Assistant,
                    content: String::new(),
                });

                let model = self.model.clone();
                let opts = ModelOptions::default()
                    .temperature(self.temperature)
                    .num_predict(self.num_predict);

                let task = match self.mode {
                    Mode::Chat => {
                        Task::stream(
                            stream_chat_oai(model, prompt, opts, self.history.clone(), self.config.clone())
                        )
                    }
                };
                Task::batch([task, snap_to(LOG.clone(), RelativeOffset::END)])
            }

            // Add to the last (added as empty) Line in self.lines.
            Message::LlmChunk(chunk) => {
                if let Some(last) = self.lines.last_mut() {
                    if matches!(last.role, Role::Assistant) {
                        trace!("{}", chunk);
                        if chunk.starts_with(&last.content) {
                            last.content = chunk;
                        } else {
                            last.content.push_str(&chunk);
                        }
                    }
                }
                snap_to(LOG.clone(), RelativeOffset::END)
            }

            Message::LlmDone => {
                self.waiting = false;
                if self.mode == Mode::Chat {
                    // Here we used to trim.
                    if let Some(last) = self.lines.last_mut() {
                        last.content.push_str("\n");
                    }
                    trace!("H: {:?}", self.history);
                }
                snap_to(LOG.clone(), RelativeOffset::END)
            }

            Message::LlmErr(e) => {
                self.waiting = false;
                if let Some(last) = self.lines.last_mut() {
                    if matches!(last.role, Role::Assistant) && last.content.is_empty() {
                        last.content = format!("(error) {e}");
                        return Task::none();
                    }
                }
                self.lines.push(Line {
                    role: Role::System,
                    content: format!("(error) {e}"),
                });
                Task::none()
            }
        }
    }

    fn view(&self) -> Element<'_, Message> {
        const MY_FONT: iced::Font = iced::Font::with_name("FiraMono Nerd Font Mono");
        
        let transcript = self.lines.iter().fold(column![].spacing(8), |col, line| {
            let prefix = match line.role {
                Role::User => "You: ",
                Role::Assistant => &self.label,
                Role::System => "",
            };
            col.push(text(format!("{prefix}{}", line.content)).size(self.config.fontsize).font(MY_FONT).line_height(LineHeight::Relative(1.4)))
        });

        let top = container(
            scrollable(container(transcript).padding(12).width(Length::Fill))
                .id(LOG.clone())
                .height(Length::Fill),
        )
        .width(Length::Fill)
        .height(Length::Fill);

        let controls = row![
            // text("Mode:").font(MY_FONT),
            // pick_list(&MODES[..], Some(self.mode), Message::ModeChanged),
            text(format!("T: {:.1}", self.temperature)).font(MY_FONT),
            slider(0.0..=1.2, self.temperature, Message::TemperatureChanged)
                .width(Length::FillPortion(1))
                .step(0.1),
            text(format!("CO: {:.1}", self.config.cut_off)).font(MY_FONT),
            slider(0.0..=4.0, self.config.cut_off, Message::CutOffChanged)
                .width(Length::FillPortion(1))
                .step(0.1),
            /*text(format!("Max tokens: {}", self.num_predict)).font(MY_FONT),
            slider(1..=4096, self.num_predict, Message::NumPredictChanged)
                .width(Length::FillPortion(1))
                .step(12),*/
            text(format!("Max CTX: {}", self.config.max_context)).font(MY_FONT),
            slider(0u32..=42u32, self.config.max_context, Message::MaxContextChanged)
                .width(Length::FillPortion(1))
                .step(1u32),
            // button(text("Reset").font(MY_FONT)).on_press(Message::ResetParams),
            button(text("Clear").font(MY_FONT)).on_press(Message::ClearAll),
        ]
        .spacing(12).align_y(iced::Alignment::Center).padding(10);

        let input = text_input("Ask your question…", &self.draft)
            .on_input(Message::DraftChanged)
            .on_submit(Message::Submit)
            .padding(10)
            .size(self.config.fontsize)
            .font(MY_FONT)
            .width(Length::Fill);

        let bottom = container(
            column![
                controls,
                row![input, text(if self.waiting { "  thinking…" } else { "" })].spacing(8),
            ]
            .spacing(8),
        )
        .width(Length::Fill)
        .padding(8);

        column![top, bottom]
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
}

// Function to convert Line to ChatMessage.
// TODO let history use ChatMessage instead.
fn line_to_chat_message(line: &Line) -> ChatMessage {
    let content = ChatMessageContent::Text(line.content.clone());
    // println!("{}", content);
    match line.role {
        Role::User => ChatMessage::User { content, name: None },
        Role::System => ChatMessage::System { content, name: None },
        Role::Assistant => ChatMessage::Assistant {
            content: Some(content),
            reasoning_content: None,
            refusal: None,
            name: None,
            audio: None,
            tool_calls: None,
        },
    }
}

fn push_vec_batch(b: &RecordBatch, out: &mut Vec<Candidate>) {
    let id_idx = b.schema().index_of("id").expect("id col");
    let a_idx = b.schema().index_of("abstract").expect("abstract");
    let t_idx = b.schema().index_of("text").expect("text");
    let d_idx = b.schema().index_of("_distance").expect("_distance");

    let ids = b.column(id_idx).as_any().downcast_ref::<StringArray>().unwrap();
    let abstracts = b.column(a_idx).as_any().downcast_ref::<StringArray>().unwrap();
    let texts = b.column(t_idx).as_any().downcast_ref::<StringArray>().unwrap();
    let dists = b.column(d_idx).as_any().downcast_ref::<Float32Array>().unwrap();

    for i in 0..b.num_rows() {
        let id = ids.value(i).to_string();
        let candidate = Candidate{
            id,
            astract: abstracts.value(i).to_string(),
            text: texts.value(i).to_string(),
            vec_dist: Some(dists.value(i)),
            fts_score: None,
        };
        trace!("{}", &candidate);
        out.push(candidate);
    }
    debug!("Pushed {} vector search results to candidates.", b.num_rows());
}

fn push_fts_batch(b: &RecordBatch, out: &mut Vec<Candidate>) {
    let id_idx = b.schema().index_of("id").expect("id col");
    let a_idx = b.schema().index_of("abstract").expect("abstract");
    let t_idx = b.schema().index_of("text").expect("text");
    let s_idx = b.schema().index_of("_score").expect("_score");

    let ids = b.column(id_idx).as_any().downcast_ref::<StringArray>().unwrap();
    let abstracts = b.column(a_idx).as_any().downcast_ref::<StringArray>().unwrap();
    let texts = b.column(t_idx).as_any().downcast_ref::<StringArray>().unwrap();
    let scores = b.column(s_idx).as_any().downcast_ref::<Float32Array>().unwrap();

    for i in 0..b.num_rows() {
        let id = ids.value(i).to_string();
        let candidate = Candidate{
            id,
            astract: abstracts.value(i).to_string(),
            text: texts.value(i).to_string(),
            vec_dist: None,
            fts_score: Some(scores.value(i)),
        };
        trace!("{}", &candidate);
        out.push(candidate);
    }
    debug!("Pushed {} full-text search results to candidates.", b.num_rows());
}

// Merge in a HashMap, merge scores and data. If an item exists in both
// fts and vec batches, it will have both scores.
fn dedupe_by_id(candidates: Vec<Candidate>) -> Vec<Candidate> {
    let mut m: HashMap<String, Candidate> = HashMap::new();
    for c in candidates {
        m.entry(c.id.clone())
            .and_modify(|e| {
                e.vec_dist = e.vec_dist.or(c.vec_dist);
                e.fts_score = e.fts_score.or(c.fts_score);
                if e.text.is_empty() { e.text = c.text.clone(); }
                if e.astract.is_empty() { e.astract = c.astract.clone(); }
            })
            .or_insert(c);
    }
    debug!("After dedup {}", m.len());
    m.into_values().collect()
}


async fn fuse_and_rerank(
    table: &lancedb::table::Table,
    qv: &[f32],
    user_prompt: &str,
    k_final: usize,
) -> anyhow::Result<Vec<Candidate>> {
    let k_candidates = k_final * 5;

    let vec_batches: Vec<RecordBatch> = table.query()
        .nearest_to(qv)?
        .limit(k_candidates)
        .refine_factor(4)
        .execute().await?
        .try_collect().await?;

    let fts = FullTextSearchQuery::new(user_prompt.to_string())
        .with_column("abstract".to_string())?;
    let fts_batches: Vec<RecordBatch> = table.query()
        .full_text_search(fts)
        .limit(k_candidates)
        .execute().await?
        .try_collect().await?;

    let mut pool = Vec::with_capacity(k_candidates * 2);
    for b in &vec_batches { push_vec_batch(b, &mut pool); }
    for b in &fts_batches { push_fts_batch(b, &mut pool); }

    let pool = dedupe_by_id(pool);

    let mut reranker = TextRerank::try_new(
        RerankInitOptions::new(RerankerModel::BGERerankerV2M3) //BGERerankerBase)
    )?;

    let passages: Vec<String> = pool.iter()
        .map(|c| if c.astract.is_empty() { c.text.clone() } else { format!("{}\n{}", c.astract, c.text) })
        .collect();

    let passages_ref: Vec<&str> = passages.iter().map(|s| s.as_str()).collect();

    let ranked = reranker.rerank(user_prompt, passages_ref.as_slice(), false, None)?;

    let mut rer_score: Vec<(usize, f32)> = ranked.into_iter().map(|r| (r.index, r.score)).collect();
    rer_score.sort_by(|a,b| b.1.total_cmp(&a.1));

    Ok(rer_score.into_iter().take(k_final).map(|(i, _)| pool[i].clone()).collect())
}

fn stream_chat_oai(
    model: String,
    user_prompt: String,
    opts: ModelOptions,
    history: Arc<Mutex<Vec<Line>>>,
    config: AppConfig,
) -> impl tokio_stream::Stream<Item = Message> + Send + 'static {
    stream! {
        let client = Client::new_from_env();
        info!("Q: {}", user_prompt);

        // Moderation ----
        let parameters = ModerationParametersBuilder::default()
                .model(ModerationModel::OmniModerationLatest.to_string())
                // .input(ModerationInput::Text("I want to kill them.".to_string()))
                .input(ModerationInput::Array(vec![
                    user_prompt.clone()
                ]))
                .build()
                .unwrap();
        let result = client.moderations().create(parameters).await.unwrap();
        let cats = &result.results[0].category_scores;
        let _flagged = result.results[0].flagged;
        
        // The moderator is very strict, we check scores.
        let mut flagged = false;
        let v = serde_json::to_value(&cats).unwrap();
        if let Value::Object(map) = v {
            for (k, v) in map {
                let score = v.as_f64().unwrap();
                if score > 0.6 {
                    debug!("{k}: {score}");
                    flagged = true;
                }
            }
        }
        
        if flagged {
            debug!("Mod: {:?}", cats);
            yield Message::LlmChunk("Please ask another question!".to_string());
            yield Message::LlmDone;
            return;
        }

        // We should check for CTX==0, and skip this if it is.
        
        let mut context = "Use the following info to answer the question, if there is none, use your own knowledge.\n".to_string();
        
        if config.max_context > 0 {
            debug!("Searching for context.");

            // Insert Db/RAG here?) //
            let table_name = config.table_name;
            let db: lancedb::Connection = {
                let guard = config.db_connexion.lock().unwrap();
                guard.clone().take().expect("Expected a database connection!")
            };

            let q = {
                let mut e = config.embedder.lock().unwrap();
                e.embed(vec![&user_prompt], None).expect("Cannot embed query?")
            };
            let qv = &q[0];

            // We need two variables for retrieval limit and for inclusion limit (after the
            // reranker. 
            // Should the first limit be twice the CTX slider, just to get some extra?
            // or should we always retrieve "many"?

            if let Ok(ref table) = db.open_table(&table_name).execute().await {
                let results_v: Vec<RecordBatch> = table
                    .query()
                    .nearest_to(qv.as_slice()).expect("err")
                    .limit(config.max_context as usize)
                    .refine_factor(4) // I pulled this number out of my hat.
                    .execute()
                    .await.expect("err")
                    .try_collect()
                    .await.expect("err");
                for b in &results_v {
                    let astractidx = b.schema().index_of("abstract").expect("err");
                    let text_idx = b.schema().index_of("text").expect("err");
                    let dist_idx = b.schema().index_of("_distance").expect("err");

                    let abstracts = b.column(astractidx).as_any().downcast_ref::<StringArray>().unwrap();
                    let texts = b.column(text_idx).as_any().downcast_ref::<StringArray>().unwrap();
                    let dists = b.column(dist_idx).as_any().downcast_ref::<Float32Array>().unwrap();

                    let (retrieved, min_dist, max_dist) =
                        (0..b.num_rows()).fold((0usize, f32::MAX, 0f32), |(cnt, min_d, max_d), i| {
                    
                            let dist = dists.value(i);
                            let astract = abstracts.value(i);
                            let text = texts.value(i);

                            let min_d = min_d.min(dist);
                            let max_d = max_d.max(dist);

                            if dist < config.cut_off {
                                debug!("{dist:.3} * {astract}: {text}");
                                context += text;
                                (cnt + 1, min_d, max_d)
                            } else {
                                debug!("{dist:.3}   {astract}: {text}");
                                (cnt, min_d, max_d)
                            }
                        });
                    info!("Retrieved {retrieved} ({:.2}-{:.2}) items.", min_dist, max_dist);
                } // for
            };
        } // max_context > 0

        let mut messages: Vec<ChatMessage> = {
            let h = history.lock().unwrap();
            h.iter().map(line_to_chat_message).collect()
        };

        messages.push(ChatMessage::User {
            content: ChatMessageContent::Text(context),
            name: None,
        });
        
        messages.push(ChatMessage::User {
            content: ChatMessageContent::Text(user_prompt.clone()),
            name: None,
        });
        debug!("{:?}", messages);

        let params = match ChatCompletionParametersBuilder::default()
            .model(model)
            .messages(messages)
            .max_tokens(opts.num_predict)
            .temperature(opts.temperature)
            .response_format(ChatCompletionResponseFormat::Text)
            .build()
        {
            Ok(p) => p,
            Err(e) => {
                yield Message::LlmErr(e.to_string());
                yield Message::LlmDone; return;
            }
        };

        let stream0 = match client.chat().create_stream(params).await {
            Ok(s) => s,
            Err(e) => { yield Message::LlmErr(e.to_string()); yield Message::LlmDone; return; }
        };

        let mut tracked = RoleTrackingStream::new(stream0);
        let mut assistant_acc = String::new();

        while let Some(item) = tracked.next().await {
            match item {
                Ok(chat_response) => {
                    for choice in &chat_response.choices {
                        if let DeltaChatMessage::Assistant { content: Some(delta), .. } = &choice.delta {
                            let s = delta.to_string();
                            if !s.is_empty() {
                                assistant_acc.push_str(&s);
                                yield Message::LlmChunk(s);
                            }
                        }
                    }
                }
                Err(e) => {
                    yield Message::LlmErr(e.to_string());
                    yield Message::LlmDone;
                    return;
                }
            }
        }

        { // Inside a block because we cannot hold the guard.
            let mut h = history.lock().unwrap();
            // info!("Q: {}", user_prompt);
            h.push(Line { role: Role::User, content: user_prompt });
            info!("A: {}", assistant_acc);
            h.push(Line { role: Role::Assistant, content: assistant_acc });
        }

        yield Message::LlmDone;
    }
}
