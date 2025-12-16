use iced::{
    executor, Application, Command, Element, Length, Settings, Theme,
    widget::{column, container, row, scrollable, text, text_input, Space},
};

pub fn main() -> iced::Result {
    EchoChat::run(Settings::default())
}

#[derive(Debug, Clone)]
enum Role { User, Assistant, System }

#[derive(Debug, Clone)]
struct Line { role: Role, content: String }

#[derive(Default)]
struct EchoChat {
    draft: String,
    lines: Vec<Line>,
    waiting: bool,
}

#[derive(Debug, Clone)]
enum Message {
    DraftChanged(String),
    Submit,
    OllamaResponse(String),
    OllamaError(String),
}

impl Application for EchoChat {
    type Executor = executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = ();

    fn new(_flags: ()) -> (Self, Command<Message>) {
        let mut s = Self::default();
        s.lines.push(Line{ role: Role::System, content: "Ready.".into() });
        (s, Command::none())
    }

    fn title(&self) -> String { "Iced 0.14 Chat Transcript".into() }

    fn update(&mut self, msg: Message) -> Command<Message> {
        match msg {
            Message::DraftChanged(s) => {
                self.draft = s;
                Command::none()
            }
            Message::Submit => {
                let user_text = self.draft.trim().to_string();
                if user_text.is_empty() || self.waiting { return Command::none(); }

                self.draft.clear();
                self.waiting = true;

                self.lines.push(Line{ role: Role::User, content: user_text.clone() });
                self.lines.push(Line{ role: Role::Assistant, content: "…".into() }); // placeholder

                // Replace this stub with an ollama-rs call later
                Command::perform(fake_llm_call(user_text), |r| match r {
                    Ok(s) => Message::OllamaResponse(s),
                    Err(e) => Message::OllamaError(e),
                })
            }
            Message::OllamaResponse(reply) => {
                self.waiting = false;
                if let Some(last) = self.lines.last_mut() {
                    if matches!(last.role, Role::Assistant) && last.content == "…" {
                        last.content = reply;
                        return Command::none();
                    }
                }
                self.lines.push(Line{ role: Role::Assistant, content: reply });
                Command::none()
            }
            Message::OllamaError(err) => {
                self.waiting = false;
                if let Some(last) = self.lines.last_mut() {
                    if matches!(last.role, Role::Assistant) && last.content == "…" {
                        last.content = format!("(error) {err}");
                        return Command::none();
                    }
                }
                self.lines.push(Line{ role: Role::System, content: format!("(error) {err}") });
                Command::none()
            }
        }
    }

    fn view(&self) -> Element<Message> {
        let transcript = self.lines.iter().fold(column![].spacing(6), |col, line| {
            let prefix = match line.role {
                Role::User => "You: ",
                Role::Assistant => "Ollama: ",
                Role::System => "",
            };
            col.push(text(format!("{prefix}{}", line.content)).size(16))
        });

        let top_pane =
            container(
                scrollable(container(transcript).padding(12).width(Length::Fill))
                    .height(Length::Fill)
            )
            .width(Length::Fill)
            .height(Length::Fill);

        let input = text_input("Type and press Enter…", &self.draft)
            .on_input(Message::DraftChanged)
            .on_submit(Message::Submit)
            .padding(10)
            .size(16)
            .width(Length::Fill);

        let bottom_pane = container(
            row![
                input,
                Space::with_width(Length::Fixed(8.0)),
                text(if self.waiting { "thinking…" } else { "" }),
            ]
        )
        .width(Length::Fill)
        .padding(8);

        column![top_pane, bottom_pane]
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
}

async fn fake_llm_call(prompt: String) -> Result<String, String> {
    Ok(format!("(stubbed) You said: {prompt}"))
}

