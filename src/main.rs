use iced::{
    Element, Length, Settings, Theme, Task,
    widget::{column, container, row, scrollable, text, text_input, Space},
};

fn theme(_: &App) -> Theme { Theme::Dark }

pub fn main() -> iced::Result {
    iced::application(App::new, App::update, App::view)
        .title("Iced 0.14 transcript")
        .theme(theme)
        .settings(Settings::default())
        .run()
}

#[derive(Debug, Clone)]
enum Role { User, Assistant, System }

#[derive(Debug, Clone)]
struct Line { role: Role, content: String }

struct App {
    draft: String,
    lines: Vec<Line>,
    waiting: bool,
}

#[derive(Debug, Clone)]
enum Message {
    DraftChanged(String),
    Submit,
    LlmOk(String),
    LlmErr(String),
}

impl App {
    fn new() -> Self {
        Self {
            draft: String::new(),
            lines: vec![Line { role: Role::System, content: "Ready.".into() }],
            waiting: false,
        }
    }

    fn update(&mut self, msg: Message) -> Task<Message> {
        match msg {
            Message::DraftChanged(s) => { self.draft = s; Task::none() }

            Message::Submit => {
                if self.waiting { return Task::none(); }
                let prompt = self.draft.trim().to_string();
                if prompt.is_empty() { return Task::none(); }

                self.draft.clear();
                self.waiting = true;

                self.lines.push(Line { role: Role::User, content: prompt.clone() });
                self.lines.push(Line { role: Role::Assistant, content: "…".into() });

                Task::perform(fake_llm_call(prompt), |r| match r {
                    Ok(s) => Message::LlmOk(s),
                    Err(e) => Message::LlmErr(e),
                })
            }

            Message::LlmOk(reply) => {
                self.waiting = false;
                if let Some(last) = self.lines.last_mut() {
                    if matches!(last.role, Role::Assistant) && last.content == "…" {
                        last.content = reply;
                        return Task::none();
                    }
                }
                self.lines.push(Line { role: Role::Assistant, content: reply });
                Task::none()
            }

            Message::LlmErr(err) => {
                self.waiting = false;
                if let Some(last) = self.lines.last_mut() {
                    if matches!(last.role, Role::Assistant) && last.content == "…" {
                        last.content = format!("(error) {err}");
                        return Task::none();
                    }
                }
                self.lines.push(Line { role: Role::System, content: format!("(error) {err}") });
                Task::none()
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

        let top = container(
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

        let bottom = container(
            row![
                input,
                Space::new().width(Length::Fixed(8.0)),
                text(if self.waiting { "thinking…" } else { "" }),
            ]
        )
        .width(Length::Fill)
        .padding(8);

        column![top, bottom].width(Length::Fill).height(Length::Fill).into()
    }
}

async fn fake_llm_call(prompt: String) -> Result<String, String> {
    Ok(format!("(stub) You said: {prompt}"))
}
