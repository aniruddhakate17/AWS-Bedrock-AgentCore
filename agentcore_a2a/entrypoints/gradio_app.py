from agentcore_a2a.ui.gradio_chat_app import GradioChatApplication

demo = GradioChatApplication().create_demo()


if __name__ == "__main__":
    demo.launch()
