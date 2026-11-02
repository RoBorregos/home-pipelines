# Home Pipelines

Training and data pipelines for the RoBorregos RoboCup@Home team. These tools prepare the vision models and language models used by the robot for each competition.

## Pipelines

| Pipeline | Location | Purpose |
| -------- | -------- | ------- |
| Object detector | [`vision/object_detector`](vision/object_detector/readme.md) | Builds a custom YOLO detector from object videos |
| Command generator | [`hri/command_interpreter/command_generator`](hri/command_interpreter/command_generator) | Generates and paraphrases GPSR and EGPSR command datasets |
| LLM fine-tuning | [`hri/command_interpreter/training_pipeline`](hri/command_interpreter/training_pipeline) | Fine-tunes the command interpretation models with Unsloth |
| Object lists | [`hri/command_interpreter/objects`](hri/command_interpreter/objects/README.md) | Generates object reference documents from images |

## Development team

| Name               | Email                                                           | Github                                   | Role                      |
| ------------------ | --------------------------------------------------------------- | ---------------------------------------- | ------------------------- |
| Gilberto Malagamba | [A00839075@tec.mx](mailto:A00839075@tec.mx)                     | [@GilMM27](https://github.com/GilMM27)   | Programming & Integration |
| Oscar Arreola      | [oscar.arreola.jr@gmail.com](mailto:oscar.arreola.jr@gmail.com) | [@Oscar-gg](https://github.com/Oscar-gg) | Programming & Integration |
| Fernando Hernandez Cantu      | [fernando_hdz_cantu@hotmail.com](mailto:fernando_hdz_cantu@hotmail.com) | [@Fernando94654](https://github.com/Fernando94654) | Programming & Integration |

## License

This project is licensed under the [MIT License](LICENSE).
