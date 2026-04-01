from search_service import handle_search_request

from bedrock_agentcore.runtime import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@app.entrypoint
def agent_invocation(payload, context):
    print("Search runtime payload:", payload)
    print("Search runtime context:", context)
    return handle_search_request(payload)


if __name__ == "__main__":
    app.run()
