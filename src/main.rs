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
use iced::widget::Scrollable;
use iced::{Font, font};
use iced::font::Family;
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
use arrow_array::RecordBatch;
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

use ollama_rs::Ollama;
use ollama_rs::generation::chat::{request::ChatMessageRequest, ChatMessage as OllamaChatMessage, MessageRole};
use ollama_rs::history::ChatHistory;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::models::ModelOptions as OllamaModelOptions;

use std::borrow::Cow;

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

    #[arg(long, help = "Font name.", default_value = "Fira")]
    fontname: String,

    #[arg(short, long, help = "Mode, openai or ollama.", default_value = "openai")]
    mode: String,

    #[arg(short, long, help = "Search mode, vector, fulltext or both.", default_value = "vector")]
    searchmode: String,

    #[arg(short = 'M', long, help = "Model name", default_value = "gpt-4.1-nano")]
    model: String,

    #[arg(short, long, help = "System prompt/info json file.")]
    promptfile: Option<String>,

    #[arg(short, long, help = "Table name.")]
    tablename: Option<String>,

    #[arg(short = 'T', long, help = "Theme.", default_value = "tokyonight")]
    themestr: String,

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

fn load_font(s: &str) -> Font {
    match s {
        "Fira" => Font {
            family: iced::font::Family::Name("Fira Mono"),
            weight: iced::font::Weight::Medium,
            stretch: iced::font::Stretch::Normal,
            style: iced::font::Style::Normal,
        },
        "Sarasa" => Font {
            family: Family::Name("Sarasa Mono SC"),
            weight: iced::font::Weight::Normal,
            stretch: iced::font::Stretch::Normal,
            style: iced::font::Style::Normal,
        },
        _ => Font { ..Default::default() }
    }
}

async fn connect_db(db_name: String) -> lancedb::Result<lancedb::Connection> {
    lancedb::connect(&db_name).execute().await
}

fn parse_theme(s: &str) -> Theme {
    match s.trim().to_ascii_lowercase().as_str() {
        "light" => Theme::Light,
        "dark" => Theme::Dark,
        "dracula" => Theme::Dracula,
        "nord" => Theme::Nord,
        "solarizedlight" => Theme::SolarizedLight,
        "solarizeddark" => Theme::SolarizedDark,
        "gruvboxlight" => Theme::GruvboxLight,
        "gruvboxdark" => Theme::GruvboxDark,
        "catppuccinlatte" => Theme::CatppuccinLatte,
        "catppuccinfrappe" => Theme::CatppuccinFrappe,
        "catppuccinmacchiato" => Theme::CatppuccinMacchiato,
        "catppuccinmocha" => Theme::CatppuccinMocha,
        "tokyonight" => Theme::TokyoNight,
        "tokyonightstorm" => Theme::TokyoNightStorm,
        "tokyonightlight" => Theme::TokyoNightLight,
        "kanagawawave" => Theme::KanagawaWave,
        "kanagawadragon" => Theme::KanagawaDragon,
        "kanagawalotus" => Theme::KanagawaLotus,
        "moonfly" => Theme::Moonfly,
        "nightfly" => Theme::Nightfly,
        "oxocarbon" => Theme::Oxocarbon,
        "ferra" => Theme::Ferra,
        _ => Theme::Light,
    }
}

fn main() -> iced::Result {
    let cli = Cli::parse();

    // Always log our crate at debug to file; CLI level only affects screen output.
    let level_filter = LevelFilter::from_str(&cli.log_level).unwrap_or(LevelFilter::Off);
    let log_spec = LogSpecification::builder()
        .module("html5ever", LevelFilter::Off)
        .module("rusty_puff", LevelFilter::Debug)
        .build();

    let duplicate = match level_filter {
        LevelFilter::Off => Duplicate::None,
        LevelFilter::Error => Duplicate::Error,
        LevelFilter::Warn => Duplicate::Warn,
        LevelFilter::Info => Duplicate::Info,
        LevelFilter::Debug => Duplicate::Debug,
        LevelFilter::Trace => Duplicate::Trace,
    };

    let _logger = Logger::with(log_spec)
        .format(log_format)
        .log_to_file(
            FileSpec::default()
                .suppress_timestamp()
                .basename("pufenbot")
                .suffix("log"),
        )
        .append()
        .duplicate_to_stderr(duplicate)
        .write_mode(WriteMode::BufferAndFlush)
        .start().expect("Logging?");
    info!("Start");

    // Check already here, so we don't run into surprises later on.
    let mode = Mode::from_str(&cli.mode).expect("Unknow mode");
    if mode == Mode::OpenAI {
        let _oaik = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY");
        let client = Client::new_from_env(); // or Client::new(api_key);
        let result = client.models();
        let rt = Runtime::new().unwrap();
        let models = rt.block_on(
            result.list()
        );
        // println!("{:?}", models);
        if let Ok(list) = models {
            for model in list.data {
                trace!(
                    // "ID: {}, Created: {:?}, Object: {}, Owned by: {}",
                    // model.id, model.created, model.object, model.owned_by
                    "ID: {}", model.id
                );
            }
        }
    }
    
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
    let _ = rt.block_on(create_empty_table(&db_name, &table_name, dim));
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

    let bytes = match cli.fontname.as_str() {
        "Fira" => std::fs::read("assets/FiraMono-Medium.ttf").unwrap(),
        "Sarasa" => std::fs::read("assets/sarasa-mono-sc-regular.ttf").unwrap(),
        _ => panic!("Unknow font specified!")
    };

    let searchmode = SearchMode::from_str(&cli.searchmode).expect("Unknow search mode");

    let app_db_path = db_name.clone();
    let app_table_name = table_name.clone();
    let app_promptfile = promptfile;
    let app_model_str = cli.model.clone();
    let app_mode_str = cli.mode.clone();
    let app_searchmode = searchmode;
    let app_fontsize = cli.fontsize;
    let app_fontname = cli.fontname.clone();
    let app_cut_off = cli.cutoff;
    let app_max_context = 12;
    let app_db_connexion = Arc::new(Mutex::new(dbc));
    let app_embedder = Arc::new(Mutex::new(embedder));
    let app_chunk_size = cli.chunksize;

    iced::application(
        move || App::new(
            app_db_path.clone(),
            app_table_name.clone(),
            app_promptfile.clone(),
            app_model_str.clone(),
            app_mode_str.clone(),
            app_searchmode,
            app_fontsize,
            app_fontname.clone(),
            app_cut_off,
            app_max_context,
            app_db_connexion.clone(),
            app_embedder.clone(),
            app_chunk_size,
        ),
        App::update,
        App::view,
        )
        .title("Speak with Pufendorf")
        .theme(parse_theme(&cli.themestr))
        // .settings(Settings::default())
        .settings(Settings {
            fonts: vec![Cow::Owned(bytes)],
            // default_font: app_font,
            ..Settings::default()
        })
        .run()
}

impl App {
    fn new(
        db_path: String,
        table_name: String,
        promptfile: String,
        model_str: String,
        mode_str: String,
        searchmode: SearchMode,
        fontsize: u32,
        fontname: String,
        cut_off: f32,
        max_context: u32,
        db_connexion: Arc<Mutex<Option<lancedb::Connection>>>,
        embedder: Arc<Mutex<TextEmbedding>>,
        chunk_size: usize,
    ) -> (Self, Task<Message>) {
        // Read the prompts from a json file.
        // Should contain a system_prompt and extra_info.
        let file_path = Path::new(&promptfile);
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

        let mode = Mode::from_str(&mode_str).expect("Unknow mode");

        (Self {
            db_path,
            table_name,
            promptfile,
            model_str,
            mode_str,
            searchmode,
            fontsize,
            fontname,
            cut_off,
            max_context,
            db_connexion,
            embedder,
            chunk_size,

            mode: mode,

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
                self.cut_off = t;
                Task::none()
            }

            Message::MaxContextChanged(t) => {
                self.max_context = t;
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

                // let model = self.model.clone();
                let model = self.model_str.clone();
                let opts = ModelOptions::default()
                    .temperature(self.temperature)
                    .num_predict(self.num_predict);

                let table_name = self.table_name.clone();
                let db_connexion = self.db_connexion.clone();
                let embedder = self.embedder.clone();
                let searchmode = self.searchmode;
                let cut_off = self.cut_off;
                let max_context = self.max_context;

                let task = match self.mode {
                    Mode::OpenAI => {
                        if self.searchmode == SearchMode::Vector {
                            Task::stream(
                                stream_chat_oai(
                                    model,
                                    prompt,
                                    opts,
                                    self.history.clone(),
                                    table_name,
                                    db_connexion,
                                    embedder,
                                    cut_off,
                                    max_context,
                                )
                            )
                        } else {
                            Task::stream(
                                stream_chat_oai_full(
                                    model,
                                    prompt,
                                    opts,
                                    self.history.clone(),
                                    table_name,
                                    db_connexion,
                                    embedder,
                                    searchmode,
                                    cut_off,
                                    max_context,
                                )
                            )
                        }
                    }
                    Mode::Ollama => {
                        Task::stream(
                            ollama_stream_chat(
                                model,
                                prompt,
                                opts,
                                self.history.clone(),
                                table_name,
                                db_connexion,
                                embedder,
                                cut_off,
                                max_context,
                            )
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
                if self.mode == Mode::OpenAI || self.mode == Mode::Ollama { // So... always...
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
        // NB weight needs to be correct!
        let my_font = load_font(&self.fontname);
        
        let transcript = self.lines.iter().fold(column![].spacing(8), |col, line| {
            let prefix = match line.role {
                Role::User => "You: ",
                Role::Assistant => &self.label,
                Role::System => "",
            };
            col.push(text(format!("{prefix}{}", line.content))
                .size(self.fontsize)
                .font(my_font)
                .line_height(LineHeight::Relative(1.4))
            )
        });

        let transcript_scroll = Scrollable::new(
            container(transcript)
                .padding(12)
                .width(Length::Fill)
        )
        .height(Length::Fill)
        .id(LOG.clone());

        let top = container(transcript_scroll)
            .width(Length::Fill)
            .height(Length::Fill);

        /*
        let top = container(
            scrollable(container(transcript).padding(12).width(Length::Fill))
                .id(LOG.clone())
                .height(Length::Fill),
        )
        .width(Length::Fill)
        .height(Length::Fill);
        */
        
        let controls = row![
            // pick_list(&MODES[..], Some(self.mode), Message::ModeChanged),
            text(format!("T: {:.1}", self.temperature)), //.font(MY_FONT),      
            slider(0.0..=1.2, self.temperature, Message::TemperatureChanged)
                .width(Length::FillPortion(1))
                .step(0.1),
            text(format!("CO: {:.1}", self.cut_off)), //.font(MY_FONT),
            slider(0.0..=4.0, self.cut_off, Message::CutOffChanged)
                .width(Length::FillPortion(1))
                .step(0.1),
            /*text(format!("Max tokens: {}", self.num_predict)).font(MY_FONT),
            slider(1..=4096, self.num_predict, Message::NumPredictChanged)
                .width(Length::FillPortion(1))
                .step(12),*/
            text(format!("Max CTX: {}", self.max_context)), //.font(MY_FONT),
            slider(0u32..=42u32, self.max_context, Message::MaxContextChanged)
                .width(Length::FillPortion(1))
                .step(1u32),
            // button(text("Reset").font(MY_FONT)).on_press(Message::ResetParams),
            button(text("Clear")).on_press(Message::ClearAll),
        ]
        .spacing(12).align_y(iced::Alignment::Center).padding(10);

        let input = text_input("Ask your question…", &self.draft)
            .on_input(Message::DraftChanged)
            .on_submit(Message::Submit)
            .padding(10)
            .size(self.fontsize)
            .font(my_font)
            .width(Length::Fill);

        let bottom = container(
            column![
                controls,
                row![input, text(if self.waiting { " pondering…" } else { "" })].spacing(8),
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

// Function to convert Line to OpenAI compatible ChatMessage.
fn line_to_chat_message(line: &Line) -> ChatMessage {
    let content = ChatMessageContent::Text(line.content.clone());
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

// Line to OllamaChatMessage so we can feed it into the Ollama API.
fn line_to_ollama(line: &Line) -> OllamaChatMessage {
    match line.role {
        Role::User => OllamaChatMessage::new(MessageRole::User, line.content.clone()),
        Role::Assistant => OllamaChatMessage::new(MessageRole::Assistant, line.content.clone()),
        Role::System => OllamaChatMessage::new(MessageRole::System, line.content.clone()),
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
                if e.text.is_empty() {
                    e.text = c.text.clone();
                }
                if e.astract.is_empty() {
                    e.astract = c.astract.clone();
                }
            })
            .or_insert(c);
    }
    debug!("After dedup {}", m.len());
    m.into_values().collect()
}


async fn _fuse_and_rerank(
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

fn stream_chat_oai_full(
    model: String,
    user_prompt: String,
    opts: ModelOptions,
    history: Arc<Mutex<Vec<Line>>>,
    table_name: String,
    db_connexion: Arc<Mutex<Option<lancedb::Connection>>>,
    embedder: Arc<Mutex<TextEmbedding>>,
    searchmode: SearchMode,
    cut_off: f32,
    max_context: u32,
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

        let mut context = "Use the following info to answer the question, if there is none, use your own knowledge.\n".to_string();
        
        if max_context > 0 {
            debug!("Searching for context.");

            // Insert Db/RAG here?) //
            let db: lancedb::Connection = {
                let guard = db_connexion.lock().unwrap();
                guard.clone().take().expect("Expected a database connection!")
            };

            let q = {
                let mut e = embedder.lock().unwrap();
                e.embed(vec![&user_prompt], None).expect("Cannot embed query?")
            };
            let qv = &q[0];

            // We need two variables for retrieval limit and for inclusion limit (after the
            // reranker. 
            // Should the first limit be twice the CTX slider, just to get some extra?
            // or should we always retrieve "many"?


            let mut results_v: Option<Vec<RecordBatch>> = None;
            let mut results_f: Option<Vec<RecordBatch>> = None;

            if let Ok(ref table) = db.open_table(&table_name).execute().await {
                if searchmode == SearchMode::Vector || searchmode == SearchMode::Both {
                    debug!("Doing vector search.");
                    results_v = Some(
                        table.query()
                            .nearest_to(qv.as_slice()).expect("err")
                            .limit(2 * max_context as usize)
                            .refine_factor(4)
                            .execute().await.expect("err")
                            .try_collect().await.expect("err")
                    );
                }

                if searchmode == SearchMode::FullText || searchmode == SearchMode::Both {
                    debug!("Doing full-text search.");
                    let fts = FullTextSearchQuery::new(user_prompt.clone())
                        .with_column("abstract".to_string()).expect("err");

                    let stream = table.query()
                        .full_text_search(fts)
                        .limit(2 * max_context as usize)
                        .execute().await.expect("err");

                    results_f = Some(stream.try_collect().await.expect("err"));
                }

                let k_final = max_context as usize;
                let k_candidates = k_final * 2;
                let mut pool: Vec<Candidate> = Vec::with_capacity(k_candidates);

                if let Some(batches) = results_v.as_ref() {
                    for b in batches {
                        push_vec_batch(b, &mut pool);
                    }
                }

                if let Some(batches) = results_f.as_ref() {
                    for b in batches {
                        push_fts_batch(b, &mut pool);
                    }
                }

                let pool = dedupe_by_id(pool);
                info!("POOL size: {}", pool.len());

                if !pool.is_empty() {
                    let mut reranker = TextRerank::try_new(
                        RerankInitOptions::new(RerankerModel::JINARerankerV1TurboEn)
                    ).expect("err");

                    let combined: Vec<String> = pool.iter()
                        .map(|c| format!("{}\n{}", c.astract, c.text))
                        .collect();

                    let ranked = reranker
                        .rerank(user_prompt.clone(), combined.as_slice(), false, None)
                        .expect("err");

                    let mut rer_score: Vec<(usize, f32)> = ranked.into_iter().map(|r| (r.index, r.score)).collect();
                    rer_score.sort_by(|a, b| b.1.total_cmp(&a.1));

                    let highest = rer_score[0].1;
                    info!("Top score {}", highest);

                    if highest >= -2.0 {
                        let delta = cut_off;
                        let top: Vec<(&Candidate, f32)> = rer_score.iter()
                            .take_while(|(_, s)| highest - *s <= delta)
                            .take(max_context as usize)
                            .map(|(i, s)| (&pool[*i], *s))
                            .collect();

                        info!("Top count {}", top.len());
                        for (candidate, s) in top {
                            context.push_str(&candidate.text);
                            debug!("TOP: ({}) {}", s, candidate);
                        }
                    } else {
                        debug!("Probably useless retrieval (highest < -2.0)");
                    }
                } else {
                    debug!("Empty pool; skipping rerank.");
                }
            }
        }
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
                // Yield errors to the stream/chat.
                yield Message::LlmErr(e.to_string());
                yield Message::LlmDone;
                return;
            }
        };

        let stream0 = match client.chat().create_stream(params).await {
            Ok(s) => s,
            Err(e) => {
                yield Message::LlmErr(e.to_string());
                yield Message::LlmDone;
                return;
            }
        };

        let mut tracked = RoleTrackingStream::new(stream0);
        let mut assistant_acc = String::new();

        while let Some(item) = tracked.next().await {
            match item {
                Ok(chat_response) => {
                    trace!("{:?}", &chat_response);
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

fn ollama_stream_chat(
    model: String,
    user_prompt: String,
    opts: ModelOptions,
    history: Arc<Mutex<Vec<Line>>>,
    table_name: String,
    db_connexion: Arc<Mutex<Option<lancedb::Connection>>>,
    embedder: Arc<Mutex<TextEmbedding>>,
    cut_off: f32,
    max_context: u32,
) -> impl tokio_stream::Stream<Item = Message> + Send + 'static {
    stream! {
        info!("Q: {}", user_prompt);

        // llama3.2:latest
        // 
        // let model = "llama3.2:latest".to_string();
        // By default, it will connect to localhost:11434
        let ollama = Ollama::default();
        let options = OllamaModelOptions::default()
            .temperature(opts.temperature)
            .repeat_penalty(2.) // from default 1.1
            .repeat_last_n(-1)
            .top_k(25) // from default 40
            .top_p(0.25) // from default 0.9
            .num_ctx(16_384)
            .num_predict(opts.num_predict as i32);


        // No moderation in Ollama.

        let mut context = format!("Question:{}\n{}", &user_prompt, "Use the following info to answer the question, if there is none, use your own knowledge.\n");
        
        if max_context > 0 {
            debug!("Searching for context.");

            // Insert Db/RAG here?) //
            let db: lancedb::Connection = {
                let guard = db_connexion.lock().unwrap();
                guard.clone().take().expect("Expected a database connection!")
            };

            let q = {
                let mut e = embedder.lock().unwrap();
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
                    .limit(max_context as usize)
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

                            if dist < cut_off {
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

                // Full-text search.
                if true {
                    // Full-text query. (Also for text field?)
                    let fts = FullTextSearchQuery::new_fuzzy(user_prompt.to_string(), None)
                        // Abstract was good for names in lucris
                        // .with_column("abstract".to_string()).expect("err");
                        .with_column("text".to_string()).expect("err");
                    let stream = table.query()
                        .full_text_search(fts)
                        .limit(2 * max_context as usize)
                        .execute()
                        .await.expect("err");

                    let results_f: Vec<arrow_array::RecordBatch> = stream.try_collect().await.expect("err");
                    for b in &results_f {
                        let astractidx = b.schema().index_of("abstract").expect("err");
                        let text_idx = b.schema().index_of("text").expect("err");
                        let dist_idx = b.schema().index_of("_score").expect("err");

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

                                // Cutoff should be done after the reranker, we take all here.
                                if true || dist < cut_off {
                                    debug!("{dist:.3} * {astract}: {text}");
                                    // context += text;
                                    (cnt + 1, min_d, max_d)
                                } else {
                                    debug!("{dist:.3}   {astract}: {text}");
                                    (cnt, min_d, max_d)
                                }
                            });
                        info!("Retrieved {retrieved} ({:.2}-{:.2}) items.", min_dist, max_dist);
                    } // for

                    let k_final = max_context as usize;
                    let k_candidates = k_final * 2;
                    let mut pool: Vec<Candidate> = Vec::with_capacity(k_candidates * 2);

                    for b in &results_v {
                        push_vec_batch(b, &mut pool);
                    }
                    for b in &results_f {
                        push_fts_batch(b, &mut pool);
                    }
                    let pool = dedupe_by_id(pool);

                    let mut reranker = TextRerank::try_new(
                           RerankInitOptions::new(RerankerModel::JINARerankerV1TurboEn)
                           //BGERerankerV2M3) //BGERerankerBase)
                        ).expect("err");

                    // Combine the abstract and text for reranking.
                    let combined: Vec<String> = pool.iter()
                        .map(|c| format!("{}\n{}", c.astract, c.text))
                        .inspect(|c| trace!("{}", c))
                        .collect();
                    let ranked = reranker.rerank(user_prompt.clone(), combined.as_slice(), false, None).expect("err");

                    let mut rer_score: Vec<(usize, f32)> =
                        ranked.into_iter()
                            .map(|r| (r.index, r.score))
                            .collect();

                    rer_score.sort_by(|a, b| b.1.total_cmp(&a.1));

                    // rer_scores are (usize, f32) where usize is an index into the pool,
                    // and the score is, uhm, the score.
                    let highest= rer_score[0].1;
                    let last = rer_score[rer_score.len() - 1]; // Should check boundaries.
                    let lowest = last.1;
                    info!("Top {} - {}", highest, lowest);
                    if highest < -2.0 { // FIXME very arbitrary... works for Pufendorf.
                        info!("Top count 0");
                        debug!("Probably useless retrieval.");
                    } else {
                        let delta = cut_off; // 1.5; // Fantasy number... larger is more context.
                        debug!("best/delta {}/{}", highest, delta);
                        let top: Vec<(&Candidate, f32)> = rer_score.iter()
                            // .take(k_final) // we don't know if we want all of them...
                            .take_while(|(_, s)| highest - *s <= delta)
                            .map(|(i, s)| (&pool[*i], *s)) // Index into pool to get &Candidate, plus the score.
                            .inspect(|r| trace!("{}", r.0)) // r is (&Candidate, f32)
                            .collect();

                        let (highest, lowest) = if !top.is_empty() {
                            let highest = top.iter().map(|(_, score)| score).fold(f32::MIN, |a, &b| a.max(b));
                            let lowest = top.iter().map(|(_, score)| score).fold(f32::MAX, |a, &b| a.min(b));
                            (highest, lowest)
                        } else {
                            debug!("No elements in top!");
                            (0.0, 0.0)
                        };

                        info!("Top count {} ({} - {})", top.len(), highest, lowest);
                        for (candidate, s) in top {
                            context += &candidate.text;
                            debug!("TOP: ({}) {}", s, candidate);
                        }
                    }
                } // if False
            };
        } // max_context > 0

        /*
        let mut stream = ollama.generate_stream(GenerationRequest::new(model.clone(), user_prompt.clone())
            .options(options)).await.unwrap();
        while let Some(res) = stream.next().await {
            let responses = res.unwrap();
            for resp in responses {
                let chunk = resp.response; //.response.as_bytes().to_string();
                yield Message::LlmChunk(chunk);
            }
        }
        // yield Message::LlmDone;
        // return;
        */

        // Get the history, convert to Ollama style ChatMessages so the API
        // accepts them. Needs to be wrapped in an Arc::Mutex.
        let ollama_history_vec: Vec<OllamaChatMessage> = {
            let h = history.lock().unwrap();
            h.iter().map(line_to_ollama).collect()
        };
        let ollama_history = Arc::new(Mutex::new(ollama_history_vec));

        // New connection for Ollama.
        // Only supply the prompt, the rest is taken care of by the send_chat_message_w(...) function.
        let req = ChatMessageRequest::new(
            model.clone(),
            vec![OllamaChatMessage::user(context.clone())] //user_prompt.clone())]
        ).options(options);

        // Prepare the streaming function.
        let mut s = match ollama.send_chat_messages_with_history_stream(ollama_history.clone(), req).await {
            Ok(s) => s,
            Err(e) => {
                yield Message::LlmErr(e.to_string());
                yield Message::LlmDone;
                return;
            }
        };

        // Collect the reply here.
        let mut assistant_acc = String::new();

        while let Some(item) = s.next().await {
            match item {
                Ok(res) => {
                    let chunk = res.message.content;
                    if !chunk.is_empty() {
                        assistant_acc.push_str(&chunk);
                        yield Message::LlmChunk(chunk);
                    }
                }
                Err(_e) => {
                    yield Message::LlmErr("Error streaming Ollama".to_string());
                    yield Message::LlmDone;
                    return;
                }
            }
        }

        // Update history with the question and answer. See oai function.
        {
            let mut h = history.lock().unwrap();
            h.push(Line { role: Role::User, content: user_prompt}); // Note, prompt w/o context.
            h.push(Line { role: Role::Assistant, content: assistant_acc });
        }

        // Debugging.
        {
            let h = ollama_history.lock().unwrap();
            for (i, msg) in h.iter().enumerate() {
                debug!("[{i}] {:?}: {}", msg.role, msg.content);
            }
        }

        yield Message::LlmDone;
    }
}

fn stream_chat_oai(
    model: String,
    user_prompt: String,
    opts: ModelOptions,
    history: Arc<Mutex<Vec<Line>>>,
    table_name: String,
    db_connexion: Arc<Mutex<Option<lancedb::Connection>>>,
    embedder: Arc<Mutex<TextEmbedding>>,
    cut_off: f32,
    max_context: u32,
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

        let mut context = "Use the following info to answer the question, if there is none, use your own knowledge.\n".to_string();
        
        if max_context > 0 {
            debug!("Searching for context.");

            // Insert Db/RAG here?) //
            let db: lancedb::Connection = {
                let guard = db_connexion.lock().unwrap();
                guard.clone().take().expect("Expected a database connection!")
            };

            let q = {
                let mut e = embedder.lock().unwrap();
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
                    .limit(max_context as usize)
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

                            if dist < cut_off {
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
                // Yield errors to the stream/chat.
                yield Message::LlmErr(e.to_string());
                yield Message::LlmDone;
                return;
            }
        };

        let stream0 = match client.chat().create_stream(params).await {
            Ok(s) => s,
            Err(e) => {
                yield Message::LlmErr(e.to_string());
                yield Message::LlmDone;
                return;
            }
        };

        let mut tracked = RoleTrackingStream::new(stream0);
        let mut assistant_acc = String::new();

        while let Some(item) = tracked.next().await {
            match item {
                Ok(chat_response) => {
                    trace!("{:?}", &chat_response);
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
