from summary_service import handle_summary_request

from bedrock_agentcore.runtime import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@app.entrypoint
def agent_invocation(payload, context):
    print("Summary runtime payload:", payload)
    print("Summary runtime context:", context)
    return handle_summary_request(payload)


if __name__ == "__main__":
    app.run()
