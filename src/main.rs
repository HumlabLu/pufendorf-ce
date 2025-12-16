use iced::widget::operation::snap_to;
use iced::widget::scrollable::RelativeOffset;
use iced::widget::Id;
use iced::{
    widget::{column, container, row, scrollable, text, text_input},
    Element, Length, Settings, Task, Theme,
};
use std::sync::LazyLock;

use async_stream::stream;
use tokio_stream::StreamExt;

use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::Ollama;

static LOG: LazyLock<Id> = LazyLock::new(|| Id::new("log"));

fn theme(_: &App) -> Theme {
    Theme::Dark
}

pub fn main() -> iced::Result {
    iced::application(App::new, App::update, App::view)
        .title("Iced 0.14 + Ollama streaming")
        .theme(theme)
        .settings(Settings::default())
        .run()
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

struct App {
    model: String,
    draft: String,
    lines: Vec<Line>,
    waiting: bool,
}

#[derive(Debug, Clone)]
enum Message {
    DraftChanged(String),
    Submit,
    LlmChunk(String),
    LlmDone,
    LlmErr(String),
}

impl App {
    fn new() -> Self {
        Self {
            model: "llama3.2:latest".into(),
            draft: String::new(),
            lines: vec![Line {
                role: Role::System,
                content: "Ready.".into(),
            }],
            waiting: false,
        }
    }

    fn update(&mut self, msg: Message) -> Task<Message> {
        match msg {
            Message::DraftChanged(s) => {
                self.draft = s;
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

                Task::batch([
                    Task::stream(ollama_stream(self.model.clone(), prompt)),
                    // scrollable::snap_to(LOG.clone(), RelativeOffset::END),
                    snap_to(LOG.clone(), RelativeOffset::END),
                ])
            }

            Message::LlmChunk(chunk) => {
                if let Some(last) = self.lines.last_mut() {
                    if matches!(last.role, Role::Assistant) {
                        last.content.push_str(&chunk);
                    }
                }
                snap_to(LOG.clone(), RelativeOffset::END)
            }

            Message::LlmDone => {
                self.waiting = false;
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

        let input = text_input("Type and press Enter…", &self.draft)
            .on_input(Message::DraftChanged)
            .on_submit(Message::Submit)
            .padding(10)
            .size(16)
            .width(Length::Fill);

        let bottom =
            container(row![input, text(if self.waiting { "  thinking…" } else { "" })].spacing(8))
                .width(Length::Fill)
                .padding(8);

        column![top, bottom]
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
}

fn ollama_stream(
    model: String,
    prompt: String,
) -> impl tokio_stream::Stream<Item = Message> + Send + 'static {
    stream! {
        let ollama = Ollama::default();

        let req = GenerationRequest::new(model, prompt);

        let mut s = match ollama.generate_stream(req).await {
            Ok(s) => s,
            Err(e) => { yield Message::LlmErr(e.to_string()); yield Message::LlmDone; return; }
        };

        while let Some(item) = s.next().await {
            match item {
                Ok(responses) => {
                    for r in responses {
                        if !r.response.is_empty() {
                            yield Message::LlmChunk(r.response);
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

        yield Message::LlmDone;
    }
}
