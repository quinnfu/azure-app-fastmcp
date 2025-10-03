from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

async def test_server():
    transport = StreamableHttpTransport("http://localhost:8000/mcp")
    async with Client(transport, auth="oauth") as client:  # Handles OAuth flow
        result = await client.call_tool("tool")
        print(result)  # Output: 8