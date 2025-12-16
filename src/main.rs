use iced::{
    executor,
    widget::{column, container, row, scrollable, text, text_input},
    Application, Element, Length, Settings, Theme,
};
use std::process::Command;

pub fn main() -> iced::Result {
    EchoApp::run(Settings::default())
}

#[derive(Default)]
struct EchoApp {
    input: String,
    echoed: String,
}

#[derive(Debug, Clone)]
enum Message {
    InputChanged(String),
    Submit,
}

impl Application for EchoApp {
    type Executor = executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = ();

    fn new(_flags: ()) -> (Self, Command<Message>) {
        (Self::default(), Command::none())
    }

    fn title(&self) -> String {
        "Iced 0.14 Two-Pane Echo".into()
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::InputChanged(s) => {
                self.input = s;
                self.echoed = self.input.clone(); // live echo
            }
            Message::Submit => {
                self.echoed = self.input.clone();
                self.input.clear();
            }
        }
        Command::none()
    }

    fn view(&self) -> Element<Message> {
        let top_pane = container(
            scrollable(
                container(text(&self.echoed).size(18))
                    .padding(12)
                    .width(Length::Fill),
            )
            .height(Length::Fill),
        )
        .width(Length::Fill)
        .height(Length::Fill);

        let bottom_pane = container(
            row![text_input("Type here…", &self.input)
                .on_input(Message::InputChanged)
                .on_submit(Message::Submit)
                .padding(10)
                .size(16)
                .width(Length::Fill)]
            .spacing(8),
        )
        .width(Length::Fill)
        .padding(8);

        column![top_pane, bottom_pane]
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
}
