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
use serde_json::Value;
use std::fs;
use std::path::Path;
use openai_dive::v1::api::Client;
use openai_dive::v1::endpoints::chat::RoleTrackingStream;
use openai_dive::v1::models::Gpt5Model;
use openai_dive::v1::resources::chat::{
    ChatCompletionParametersBuilder, ChatCompletionResponseFormat, ChatMessage, ChatMessageContent,
    DeltaChatMessage,
};

// LOG is the Id for the chat log output pane, needed in the snap_to(...) function.
static LOG: LazyLock<Id> = LazyLock::new(|| Id::new("log"));
static MODES: [Mode; 1] = [Mode::Chat];

fn theme(_: &App) -> Theme {
    Theme::Dark
}

// ----
// Struct for model options like temp &c.
#[derive(Debug, Default, Clone)]
pub struct ModelOptions {
    pub temperature: f32,
    pub num_predict: i32,
}
impl ModelOptions {
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn num_predict(mut self, num_predict: i32) -> Self {
        self.num_predict = num_predict;
        self
    }
}
// ----

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Chat,
}
impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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

// Messages for iced GUI.
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

// "Global" data for the iced app.
struct App {
    model: String,
    mode: Mode,

    temperature: f32,
    num_predict: i32,
    max_turns: u16,

    draft: String, // User input
    lines: Vec<Line>,
    waiting: bool,

    history: Arc<Mutex<Vec<Line>>>,

    system_prompt: String,
    extra_info: String,
}

// use futures::StreamExt;

#[tokio::main]
async fn get_models() {
    let client = Client::new_from_env();
    let result = client
        .models()
        .list()
        .await.unwrap();
    for model in result.data {
        println!("{}", model.id);
    }
}

#[tokio::main]
async fn openai_stream() {
    let client = Client::new_from_env();

    let parameters = ChatCompletionParametersBuilder::default()
        .model(Gpt5Model::Gpt5Nano.to_string())
        .messages(vec![
            ChatMessage::User {
                content: ChatMessageContent::Text("Hello!".to_string()),
                name: None,
            },
            ChatMessage::User {
                content: ChatMessageContent::Text(
                    "What are the five biggest cities in Vietnam?".to_string(),
                ),
                name: None,
            },
        ])
        .response_format(ChatCompletionResponseFormat::Text)
        .build()
        .unwrap();

    let stream = client.chat().create_stream(parameters).await.unwrap();

    // The stream will receive a chunk of a chat completion response. The first chunk will contain the role of the message, and subsequent chunks won't contain the role.
    // When a chunk is received without a role, it will use the `DeltaChatMessage::Untagged` variant. And you'll have to manually infer the role of the message based on the previous messages.
    // The 'RoleTrackingStream' is a wrapper around the stream that does this for you - it will automatically infer the role of the message and return the correct variant.

    let mut tracked_stream = RoleTrackingStream::new(stream);

    while let Some(response) = tracked_stream.next().await {
        match response {
            Ok(chat_response) => {
                chat_response
                    .choices
                    .iter()
                    .for_each(|choice| match &choice.delta {
                        DeltaChatMessage::User { content, .. } => {
                            print!("{content}");
                        }
                        DeltaChatMessage::System { content, .. } => {
                            print!("{content}");
                        }
                        DeltaChatMessage::Assistant {
                            content: Some(chat_message_content),
                            ..
                        } => {
                            print!("{chat_message_content}");
                        }
                        _ => {}
                    })
            }
            Err(e) => eprintln!("{e}"),
        }
    }
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
    // get_models();
    // openai_stream();

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

        let history = Arc::new(Mutex::new(vec![
            Line {
                role: Role::System,
                content: sysprompt.clone(),
            }]
        ));
        /*
        pub enum Gpt5Model {
            #[serde(rename = "gpt-5")]
            Gpt5,
            #[serde(rename = "gpt-5-mini")]
            Gpt5Mini,
            #[serde(rename = "gpt-5-nano")]
            Gpt5Nano,
        }
        #[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
        pub enum Gpt4Model {
            #[serde(rename = "gpt-4.1")]
            Gpt41,
            #[serde(rename = "gpt-4o")]
            Gpt4O,
            #[serde(rename = "gpt-4o-audio-preview")]
            Gpt4OAudioPreview,
        }
        https://community.openai.com/t/is-anyone-experiencing-issues-with-gpt-5-nano-returning-no-output/1351246
        */
        Self {
            // gpt-5-nano
            model: "gpt-4.1-nano".into(), //Gpt5Model::Gpt5Nano.to_string()
            mode: Mode::Chat,

            temperature: 0.05,
            num_predict: 1024,
            max_turns: 20,

            // Draft is user input, lines are everyting inthe output pane.
            draft: String::new(),
            lines: vec![Line {
                role: Role::System,
                content: "Ready.".into(),
            }],
            waiting: false,

            history,

            system_prompt: sysprompt,
            extra_info: "The year is 1667".into(), // Not used.
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
                self.num_predict = 512;
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
                *self.history.lock().unwrap() = vec![Line {
                    role: Role::System,
                    content: "You are a helpful assistant.".to_string(),
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
                        Task::stream(stream_chat_oai(model, prompt, opts, self.history.clone()))
                    }
                };
                // Task::none()
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
            
            // Add to the last (added as empty) Line in self.lines.
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
                    // Here we used to trim.
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
                Role::Assistant => "Pufendorf: ",
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
/*
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
*/
fn _stream_chat(
    _model: String,
    _user_prompt: String,
    _opts: ModelOptions,
    _history: Arc<Mutex<Vec<Line>>>,
) -> impl tokio_stream::Stream<Item = Message> + Send + 'static {
    stream! {

        let txt = vec!["Some", "words"];
        for t in txt {
            let chunk = t;
            if !chunk.is_empty() {
                yield Message::LlmChunk(chunk.to_string());
                yield Message::LlmChunk(" ".to_string());
            }
        }

        yield Message::LlmDone;
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

fn stream_chat_oai(
    model: String,
    user_prompt: String,
    opts: ModelOptions,
    history: Arc<Mutex<Vec<Line>>>,
) -> impl tokio_stream::Stream<Item = Message> + Send + 'static {
    stream! {
        let client = Client::new_from_env();

        let mut messages: Vec<ChatMessage> = {
            let h = history.lock().unwrap();
            h.iter().map(line_to_chat_message).collect()
        };

        messages.push(ChatMessage::User {
            content: ChatMessageContent::Text(user_prompt.clone()),
            name: None,
        });

        let max_tokens = u32::try_from(opts.num_predict).ok();
        let params = match ChatCompletionParametersBuilder::default()
            .model(model)
            .messages(messages)
            .max_tokens(max_tokens.unwrap())
            .temperature(opts.temperature)
            .response_format(ChatCompletionResponseFormat::Text)
            .build()
        {
            Ok(p) => p,
            Err(e) => { yield Message::LlmErr(e.to_string()); yield Message::LlmDone; return; }
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
                Err(e) => { yield Message::LlmErr(e.to_string()); yield Message::LlmDone; return; }
            }
        }

        {
            let mut h = history.lock().unwrap();
            h.push(Line { role: Role::User, content: user_prompt });
            // println!("Pusing: {}", &assistant_acc);
            h.push(Line { role: Role::Assistant, content: assistant_acc });
        }

        yield Message::LlmDone;
    }
}
