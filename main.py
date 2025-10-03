"""Azure (Microsoft Entra) OAuth server example for FastMCP.

Using AzureProvider (recommended approach)

Known issue: AzureProvider sends 'resource' parameter which is not supported by Azure AD v2.0 endpoint.
Temporary solution: Manually remove the resource parameter from the OAuth authorization URL in the browser.

Required environment variables:
- AZURE_CLIENT_ID: Your Azure application (client) ID
- AZURE_CLIENT_SECRET: Your Azure client secret  
- AZURE_TENANT_ID: Tenant ID

To run:
    python application.py
"""

import os
from dotenv import load_dotenv  
import httpx
import uvicorn

load_dotenv() 

from fastmcp import FastMCP
from fastmcp.server.auth.providers.azure import AzureProvider
from fastmcp.server.dependencies import get_http_request

# temporary solution to force v2.0 behavior of AzureProvider
class PatchedAzureProvider(AzureProvider):
    def _get_resource_url(self, mcp_path):
        return None  # Force v2.0 behavior

    def authorize(self, *args, **kwargs):
        # Remove resource from auth_params if present
        if len(args) >= 2 and hasattr(args[1], 'resource'):
            # Create a copy of auth_params without the resource attribute
            auth_params = args[1]
            if hasattr(auth_params, '__dict__'):
                # Create new auth_params without resource
                new_auth_params = type(auth_params)(**{k: v for k, v in auth_params.__dict__.items() if k != 'resource'})
                args = (args[0], new_auth_params) + args[2:]

        return super().authorize(*args, **kwargs)


auth = PatchedAzureProvider(
    client_id=os.getenv("AZURE_CLIENT_ID") or "",
    client_secret=os.getenv("AZURE_CLIENT_SECRET") or "",
    tenant_id=os.getenv("AZURE_TENANT_ID") or "",
    base_url=os.getenv("BASE_URL", "http://localhost:8000"),  # 使用环境变量
    # Other available parameters:
    # redirect_path="/auth/callback",  # OAuth callback path
    required_scopes=["User.Read", "email", "openid", "profile"],  # Scopes required for API access
    # allowed_client_redirect_uris=[...],  # Allowed client redirect URIs
)

mcp = FastMCP("Azure OAuth Example Server", auth=auth)


@mcp.tool
def echo(message: str) -> str:
    """Echo the provided message back to the caller."""
    return message


@mcp.tool  
def get_server_info() -> str:
    """Get server information."""
    return "FastMCP server with Azure OAuth authentication"


@mcp.tool
async def get_user_profile() -> dict:
    """Get the current logged-in user's profile information.
    
    Returns a dictionary containing the following information:
    - id: User ID
    - displayName: Display name
    - givenName: First name
    - surname: Last name
    - userPrincipalName: User principal name (usually email)
    - mail: Email address
    - jobTitle: Job title
    - officeLocation: Office location
    - mobilePhone: Mobile phone number
    - businessPhones: Business phone numbers
    - preferredLanguage: Preferred language
    """
    try:
        # Get current HTTP request
        request = get_http_request()
        
        # Get access token from request headers
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return {"error": "No valid access token provided"}
        
        access_token = auth_header.split("Bearer ")[1]
        
        # Call Microsoft Graph API to get user information
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://graph.microsoft.com/v1.0/me",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code != 200:
                return {
                    "error": f"API call failed: {response.status_code}",
                    "details": response.text
                }
            
            data = response.json()
            
            # Extract main user information
            profile = {
                "id": data.get("id"),
                "displayName": data.get("displayName"),
                "givenName": data.get("givenName"),
                "surname": data.get("surname"),
                "userPrincipalName": data.get("userPrincipalName"),
                "mail": data.get("mail"),
                "jobTitle": data.get("jobTitle"),
                "officeLocation": data.get("officeLocation"),
                "mobilePhone": data.get("mobilePhone"),
                "businessPhones": data.get("businessPhones", []),
                "preferredLanguage": data.get("preferredLanguage"),
            }
            
            return profile
            
    except Exception as e:
        return {"error": f"Failed to get user profile: {str(e)}"}

# fastmcp_asgi_app = mcp.asgi()

if __name__ == "__main__":
#     # Listen on 0.0.0.0 for all interfaces to avoid localhost/127.0.0.1 resolution issues
    mcp.run(transport="http", port=8000, host="0.0.0.0")
#     uvicorn.run('main:fastmcp_asgi_app', host='0.0.0.0', port=8000)
