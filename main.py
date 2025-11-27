# server.py
from fastmcp import FastMCP
from fastmcp.server.auth.providers.azure import AzureProvider
from fastmcp.server.dependencies import get_access_token
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Azure OAuth configuration
# ============================================================================
auth_provider = AzureProvider(
    client_id=os.getenv("AZURE_CLIENT_ID"),
    client_secret=os.getenv("AZURE_CLIENT_SECRET"),
    tenant_id=os.getenv("AZURE_TENANT_ID"),
    base_url="http://localhost:8000",
    required_scopes=["mcp_access"],  # your custom API scope
    additional_authorize_scopes=["User.Read", "Files.Read", "openid", "email", "profile"],
)

mcp = FastMCP(name="Azure Secured App", auth=auth_provider)

# ============================================================================
# Helper function: OBO flow to get Microsoft Graph API Token
# ============================================================================
async def get_graph_token() -> str | dict:
    """get Graph API token using On-Behalf-Of (OBO) flow"""
    try:
        token = get_access_token()
        token_url = f"https://login.microsoftonline.com/{os.getenv('AZURE_TENANT_ID')}/oauth2/v2.0/token"
        
        data = {
            'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
            'client_id': os.getenv('AZURE_CLIENT_ID'),
            'client_secret': os.getenv('AZURE_CLIENT_SECRET'),
            'assertion': token.token,  # user's current token
            'scope': 'https://graph.microsoft.com/.default',  # request all authorized Graph permissions
            'requested_token_use': 'on_behalf_of'
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(token_url, data=data)
            if response.status_code != 200:
                return {"error": f"Failed to get Graph token: {response.status_code}"}
            return response.json()['access_token']
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# MCP tools
# ============================================================================

@mcp.tool
async def get_user_info() -> dict:
    """get basic user information"""
    token = get_access_token()
    return {
        "name": token.claims.get("name"),
        "email": token.claims.get("preferred_username"),
        "azure_id": token.claims.get("sub"),
    }

@mcp.tool
async def get_onedrive_info() -> dict:
    """get user's OneDrive information"""
    try:
        # step 1: get Graph API token
        graph_token = await get_graph_token()
        if isinstance(graph_token, dict):  # error
            return graph_token
        
        # step 2: call Microsoft Graph API
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://graph.microsoft.com/v1.0/me/drive",
                headers={"Authorization": f"Bearer {graph_token}"}
            )
            
            if response.status_code != 200:
                return {"error": f"Graph API error: {response.status_code}"}
            
            return response.json()
    except Exception as e:
        return {"error": str(e)}