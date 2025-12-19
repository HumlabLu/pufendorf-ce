use iced::widget::operation::snap_to;
use iced::widget::scrollable::RelativeOffset;
use iced::widget::Id;

use iced::{
    widget::{button, column, container, pick_list, row, scrollable, slider, text, text_input},
    Element, Length, Settings, Task, Theme,
};
use std::{
    fmt,
    sync::{Arc, LazyLock, Mutex},
};

use async_stream::stream;
use tokio_stream::StreamExt;

use bm25::{Document, Language, SearchEngineBuilder, SearchResult};
use ollama_rs::generation::chat::MessageRole;
use ollama_rs::{
    generation::chat::{request::ChatMessageRequest, ChatMessage},
    generation::completion::request::GenerationRequest,
    models::ModelOptions,
    Ollama,
};
use serde_json::Value;
use std::fs;
use std::path::Path;

// LOG is the Id for the output pane, needed in the snap_to(...) function.
static LOG: LazyLock<Id> = LazyLock::new(|| Id::new("log"));
static MODES: [Mode; 2] = [Mode::Completion, Mode::Chat];

fn theme(_: &App) -> Theme {
    Theme::Dark
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Completion,
    Chat,
}
impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Mode::Completion => write!(f, "Completion"),
            Mode::Chat => write!(f, "Chat (history)"),
        }
    }
}

#[derive(Debug, Clone)]
enum Role {
    User,
    Assistant,
    System,
}
#[derive(Debug, Clone)]
struct Line {
    role: Role,
    content: String,
}

#[derive(Debug, Clone)]
enum Message {
    DraftChanged(String),
    Submit,
    ModeChanged(Mode),

    TemperatureChanged(f32),
    NumPredictChanged(i32),
    MaxTurnsChanged(u16),

    ResetParams,
    ClearAll,

    LlmChunk(String),
    LlmDone,
    LlmErr(String),
}

// "Global" data for the app.
struct App {
    model: String,
    mode: Mode,

    temperature: f32,
    num_predict: i32,
    max_turns: u16,

    draft: String, // User input
    lines: Vec<Line>,
    waiting: bool,

    history: Arc<Mutex<Vec<ChatMessage>>>,

    system_prompt: String,
    extra_info: String,
}

pub fn main() -> iced::Result {
    /*
    let file_path = Path::new("chatprompts.json");
    // Read the entire content of the JSON file into a string
    let content = fs::read_to_string(file_path).expect("no file");
    // Print just to confirm file reading success
    // println!("File Content:\n{}", content);
    let data: Value = serde_json::from_str(&content).expect("data");
    // println!("{}", &data);
    let sysprompt = &data["system_prompt"]
        .as_str()
        .unwrap_or("You are Samuel Von Pufendorf.");
    let mut sysprompt = sysprompt.to_string();
    // dbg!(sysprompt);
    if let Some(extras) = data["extra_info"].as_array() {
        for extra in extras {
            // println!("{}", extra);
            sysprompt += extra.as_str().unwrap_or("");
        }
    }
    */
    let corpus = [
        "The rabbit munched the orange carrot.",
        "The snake hugged the green lizard.",
        "The hedgehog impaled the orange orange.",
        "The squirrel buried the brown nut.",
    ];

    let search_engine = SearchEngineBuilder::<u32>::with_corpus(Language::English, corpus).build();

    let limit = 3;
    let search_results = search_engine.search("orange", limit);
    println!("{:?}", search_results);
    let search_results = search_engine.search("When were you born?", limit);
    println!("{:?}", search_results);

    iced::application(App::new, App::update, App::view)
        .title("Speak with Pufendorf")
        .theme(theme)
        .settings(Settings::default())
        .run()
}

impl App {
    fn new() -> Self {
        // Read the prompts from a json file.
        // Should contain a system_prompt and extra_info.
        let file_path = Path::new("chatprompts.json");
        let content = fs::read_to_string(file_path).expect("no file");
        let data: Value = serde_json::from_str(&content).expect("data");
        let sysprompt = &data["system_prompt"]
            .as_str()
            .unwrap_or("You are Samuel Von Pufendorf.");
        let mut sysprompt = sysprompt.to_string();
        if let Some(extras) = data["extra_info"].as_array() {
            for extra in extras {
                sysprompt += extra.as_str().unwrap_or("");
            }
        }

        let history = Arc::new(Mutex::new(vec![ChatMessage::system(
            // "You are Samuel von Pufendorf".to_string(),
            sysprompt.clone(),
        )]));

        Self {
            model: "llama3.2:latest".into(),
            mode: Mode::Chat,

            temperature: 0.1,
            num_predict: 512,
            max_turns: 20,

            draft: String::new(),
            lines: vec![Line {
                role: Role::System,
                content: "Ready.".into(),
            }],
            waiting: false,

            history,

            system_prompt: sysprompt,
            extra_info: "The year is 1667".into(),
        }
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

            Message::NumPredictChanged(n) => {
                self.num_predict = n;
                Task::none()
            }

            Message::MaxTurnsChanged(n) => {
                self.max_turns = n;
                Task::none()
            }

            Message::ResetParams => {
                self.temperature = 0.7;
                self.num_predict = 256;
                self.max_turns = 20;
                Task::none()
            }

            Message::ClearAll => {
                self.draft.clear();
                self.waiting = false;
                self.lines = vec![Line {
                    role: Role::System,
                    content: "Cleared.".into(),
                }];
                *self.history.lock().unwrap() = vec![ChatMessage::system(
                    // "You are a helpful assistant.".to_string(),
                    self.system_prompt.clone(),
                )];
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
                    Mode::Completion => Task::stream(stream_completion(model, prompt, opts)),
                    Mode::Chat => {
                        Task::stream(stream_chat(model, prompt, opts, self.history.clone()))
                    }
                };

                Task::batch([task, snap_to(LOG.clone(), RelativeOffset::END)])
            }

            /*Message::LlmChunk(chunk) => {
                if let Some(last) = self.lines.last_mut() {
                    if matches!(last.role, Role::Assistant) {
                        last.content.push_str(&chunk);
                    }
                }
                snap_to(LOG.clone(), RelativeOffset::END)
            }*/
            Message::LlmChunk(chunk) => {
                if let Some(last) = self.lines.last_mut() {
                    if matches!(last.role, Role::Assistant) {
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
                    trim_history_by_turns(
                        &mut self.history.lock().unwrap(),
                        self.max_turns as usize,
                    );
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
        let transcript = self.lines.iter().fold(column![].spacing(6), |col, line| {
            let prefix = match line.role {
                Role::User => "You: ",
                Role::Assistant => "Ollama: ",
                Role::System => "",
            };
            col.push(text(format!("{prefix}{}", line.content)).size(16))
        });

        let top = container(
            scrollable(container(transcript).padding(12).width(Length::Fill))
                .id(LOG.clone())
                .height(Length::Fill),
        )
        .width(Length::Fill)
        .height(Length::Fill);

        let _modes = [Mode::Completion, Mode::Chat];

        let controls = row![
            text("Mode:"),
            pick_list(&MODES[..], Some(self.mode), Message::ModeChanged),
            text(format!("Temp: {:.1}", self.temperature)),
            slider(0.0..=2.0, self.temperature, Message::TemperatureChanged)
                .width(Length::Fixed(180.0))
                .step(0.1),
            text(format!("Max tokens: {}", self.num_predict)),
            slider(1..=4096, self.num_predict, Message::NumPredictChanged)
                .width(Length::Fixed(180.0))
                .step(12),
            text(format!("Max turns: {}", self.max_turns)),
            slider(1u16..=100u16, self.max_turns, Message::MaxTurnsChanged)
                .width(Length::Fixed(160.0)),
            button("Reset").on_press(Message::ResetParams),
            button("Clear").on_press(Message::ClearAll),
        ]
        .spacing(12);

        let input = text_input("Type and press Enter…", &self.draft)
            .on_input(Message::DraftChanged)
            .on_submit(Message::Submit)
            .padding(10)
            .size(16)
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

fn stream_completion(
    model: String,
    prompt: String,
    opts: ModelOptions,
) -> impl tokio_stream::Stream<Item = Message> + Send + 'static {
    stream! {
        let ollama = Ollama::default();
        let req = GenerationRequest::new(model, prompt).options(opts);

        let mut s = match ollama.generate_stream(req).await {
            Ok(s) => s,
            Err(e) => { yield Message::LlmErr(e.to_string()); yield Message::LlmDone; return; }
        };

        while let Some(item) = s.next().await {
            match item {
                Ok(responses) => for r in responses { if !r.response.is_empty() { yield Message::LlmChunk(r.response); } }
                Err(e) => { yield Message::LlmErr(e.to_string()); yield Message::LlmDone; return; }
            }
        }
        yield Message::LlmDone;
    }
}

fn stream_chat(
    model: String,
    user_prompt: String,
    opts: ModelOptions,
    history: Arc<Mutex<Vec<ChatMessage>>>,
) -> impl tokio_stream::Stream<Item = Message> + Send + 'static {
    stream! {
        let ollama = Ollama::default();
        let req = ChatMessageRequest::new(model, vec![ChatMessage::user(user_prompt)]).options(opts);

        let mut s = match ollama.send_chat_messages_with_history_stream(history, req).await {
            Ok(s) => s,
            Err(e) => { yield Message::LlmErr(e.to_string()); yield Message::LlmDone; return; }
        };

        while let Some(item) = s.next().await {
            match item {
                Ok(res) => {
                    let chunk = res.message.content;
                    if !chunk.is_empty() { yield Message::LlmChunk(chunk); }
                }
                Err(_) => { yield Message::LlmErr("Ollama stream error".into()); yield Message::LlmDone; return; }
            }
        }

        yield Message::LlmDone;
    }
}

fn trim_history_by_turns(history: &mut Vec<ChatMessage>, max_turns: usize) {
    if max_turns == 0 {
        return;
    }

    let (sys, rest) = if matches!(history.first(), Some(m) if matches!(m.role, MessageRole::System))
    {
        (Some(history[0].clone()), history[1..].to_vec())
    } else {
        (None, history.clone())
    };

    let keep = max_turns.saturating_mul(2);
    let rest = if rest.len() > keep {
        rest[rest.len() - keep..].to_vec()
    } else {
        rest
    };

    history.clear();
    if let Some(s) = sys {
        history.push(s);
    }
    history.extend(rest);
}
