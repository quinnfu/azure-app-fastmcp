"""
FastMCP Client + DeepSeek LLM - Complete OAuth MCP Flow Demo
============================================================

This example demonstrates the complete OAuth 2.0 + PKCE flow with MCP:
- Detailed HTTP requests and responses at each step
- OAuth metadata discovery (RFC 9728, RFC 8414)
- PKCE parameter generation
- Token exchange with real HTTP examples
- MCP protocol operations with Bearer token authentication
- LLM integration with intelligent tool calling

Architecture:
   ┌─────────────────────────────────────────┐
   │    LLM Application (This File)          │
   ├─────────────────────────────────────────┤
   │         MCP Client (FastMCP)            │
   ├─────────────────────────────────────────┤
   │    Transport (HTTP with OAuth)          │
   ├─────────────────────────────────────────┤
   │         MCP Server (FastMCP)            │
   ├─────────────────────────────────────────┤
   │   Tools/Resources (Graph API)           │
   └─────────────────────────────────────────┘

Prerequisites:
pip install fastmcp httpx openai

Environment:
DEEPSEEK_API_KEY=your_api_key (optional for demo mode)
"""

import asyncio
import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# FastMCP imports
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# Use OpenAI-compatible client for DeepSeek
# DeepSeek provides OpenAI-compatible API
import httpx


# ============================================================================
# Data Structure Definitions
# ============================================================================

@dataclass
class Message:
    """Chat message structure"""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None  # Tool name (for tool role)
    tool_call_id: Optional[str] = None  # Tool call ID


@dataclass
class ToolCall:
    """Tool call request"""
    id: str
    name: str
    arguments: Dict[str, Any]


# ============================================================================
# DeepSeek LLM Client
# ============================================================================

class DeepSeekClient:
    """
    DeepSeek LLM Client
    
    OpenAI-compatible API supporting:
    - Chat Completions
    - Function Calling
    - Streaming
    
    API Docs: https://platform.deepseek.com/api-docs/
    """
    
    def __init__(self, api_key: str):
        """Initialize DeepSeek client with API key"""
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        self.client = httpx.AsyncClient(
            timeout=60.0,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        )
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Call DeepSeek Chat Completion API with Function Calling support"""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
        
        return response.json()
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


# ============================================================================
# MCP Client Wrapper
# ============================================================================

class MCPClientWrapper:
    """
    MCP Client Wrapper with Detailed OAuth Flow Visualization
    
    This class wraps FastMCP Client and provides:
    - Automatic connection and initialization
    - Step-by-step OAuth flow with real HTTP requests/responses
    - Tool discovery and caching
    - Simplified tool calling with Bearer token authentication
    - Comprehensive error handling
    """
    
    def __init__(self, server_url: str, auth: Optional[str] = None):
        """Initialize MCP Client Wrapper"""
        self.server_url = server_url
        self.auth = auth
        self.client: Optional[Client] = None
        self.tools_cache: List[Any] = []
        self.resources_cache: List[Any] = []
    
    async def _discover_oauth_metadata(self):
        """
        STEP 1: OAuth Metadata Discovery (RFC 9728, RFC 8414)
        
        MCP OAuth requires servers to expose metadata endpoints for automatic discovery:
        - /.well-known/oauth-protected-resource (RFC 9728)
        - /.well-known/oauth-authorization-server (RFC 8414)
        
        This allows clients to discover:
        - Authorization endpoint
        - Token endpoint  
        - Supported scopes
        - Other OAuth capabilities
        """
        print("\n" + "="*80)
        print("STEP 1: OAuth Metadata Discovery")
        print("="*80)
        print("""
Before OAuth flow begins, client discovers server's OAuth configuration.
This is standardized by RFC 9728 and RFC 8414.

Purpose:
  • Discover OAuth endpoints automatically (no hardcoding)
  • Learn supported features and capabilities
  • Understand available scopes and permissions
  • Get token endpoint and authorization URLs

Benefits:
  • Client works with any compliant OAuth server
  • No manual configuration needed
  • Future-proof against server URL changes
        """)
        
        base_url = self.server_url.rsplit('/mcp', 1)[0] if '/mcp' in self.server_url else self.server_url
        
        print(f"\n🔍 Querying Well-Known Endpoints:")
        print(f"   Base URL: {base_url}")
        
        metadata_urls = [
            (f"{base_url}/.well-known/oauth-protected-resource", "RFC 9728 - Protected Resource Metadata"),
            (f"{base_url}/.well-known/oauth-authorization-server", "RFC 8414 - Authorization Server Metadata")
        ]
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for url, rfc in metadata_urls:
                try:
                    print(f"\n{'═'*80}")
                    print(f"📡 Metadata Discovery: {rfc}")
                    print(f"{'═'*80}")
                    
                    print(f"\n📤 HTTP REQUEST")
                    print(f"{'─'*80}")
                    print(f"GET {url}")
                    print(f"Host: {base_url.split('://')[1] if '://' in base_url else base_url}")
                    print(f"Accept: application/json")
                    print(f"User-Agent: fastmcp-client/0.1.0")
                    print(f"{'─'*80}")
                    
                    response = await client.get(url)
                    
                    print(f"\n📥 HTTP RESPONSE")
                    print(f"{'─'*80}")
                    print(f"HTTP/1.1 {response.status_code} {response.reason_phrase}")
                    print(f"Content-Type: {response.headers.get('content-type', 'N/A')}")
                    print(f"Content-Length: {response.headers.get('content-length', 'N/A')}")
                    print(f"Cache-Control: {response.headers.get('cache-control', 'N/A')}")
                    print(f"")
                    
                    if response.status_code == 200:
                        metadata = response.json()
                        print(f"Response Body:")
                        print(json.dumps(metadata, indent=2, ensure_ascii=False))
                        print(f"{'─'*80}")
                        
                        if "authorization_endpoint" in metadata:
                            print(f"\n✅ OAuth Endpoints Discovered:")
                            print(f"┌──────────────────────────────────────────────────────────────┐")
                            print(f"│ Endpoint Type        │ URL                                   │")
                            print(f"├──────────────────────────────────────────────────────────────┤")
                            print(f"│ Authorization        │ {metadata.get('authorization_endpoint', 'N/A')[:40]:40} │")
                            print(f"│ Token Exchange       │ {metadata.get('token_endpoint', 'N/A')[:40]:40} │")
                            if metadata.get('registration_endpoint'):
                                print(f"│ Client Registration  │ {metadata.get('registration_endpoint', 'N/A')[:40]:40} │")
                            if metadata.get('userinfo_endpoint'):
                                print(f"│ User Info            │ {metadata.get('userinfo_endpoint', 'N/A')[:40]:40} │")
                            print(f"└──────────────────────────────────────────────────────────────┘")
                            
                        if "scopes_supported" in metadata:
                            print(f"\n✅ Supported OAuth Scopes:")
                            for scope in metadata.get('scopes_supported', []):
                                scope_descriptions = {
                                    "User.Read": "Read user's basic profile information",
                                    "Files.Read": "Read user's OneDrive files",
                                    "email": "Access user's email address",
                                    "openid": "OpenID Connect authentication",
                                    "profile": "Access user's profile information",
                                    "offline_access": "Maintain access when user is offline"
                                }
                                desc = scope_descriptions.get(scope, "No description available")
                                print(f"   • {scope:20} - {desc}")
                        
                        if "response_types_supported" in metadata:
                            print(f"\n✅ Supported Response Types:")
                            for rt in metadata.get('response_types_supported', []):
                                print(f"   • {rt}")
                        
                        if "grant_types_supported" in metadata:
                            print(f"\n✅ Supported Grant Types:")
                            for gt in metadata.get('grant_types_supported', []):
                                print(f"   • {gt}")
                        
                        if "code_challenge_methods_supported" in metadata:
                            print(f"\n✅ PKCE Support:")
                            methods = metadata.get('code_challenge_methods_supported', [])
                            if 'S256' in methods:
                                print(f"   • S256 (SHA-256) - ✓ Recommended method supported")
                            if 'plain' in methods:
                                print(f"   • plain - ⚠️  Less secure, not recommended")
                    else:
                        print(f"Response Body:")
                        print(response.text[:500])
                        print(f"{'─'*80}")
                        print(f"\n⚠️  Metadata not available at this endpoint")
                        print(f"   This may be normal - not all servers implement both metadata endpoints")
                        
                except httpx.TimeoutException:
                    print(f"\n❌ Timeout: Server did not respond within 10 seconds")
                    print(f"   Possible causes:")
                    print(f"   • MCP server not running")
                    print(f"   • Firewall blocking connection")
                    print(f"   • Server is slow to respond")
                except httpx.ConnectError:
                    print(f"\n❌ Connection Error: Cannot connect to {base_url}")
                    print(f"   Possible causes:")
                    print(f"   • MCP server not running (try: python main.py)")
                    print(f"   • Wrong URL or port")
                    print(f"   • Network connectivity issues")
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    print(f"   Type: {type(e).__name__}")
        
        print(f"\n{'═'*80}")
        print("✅ Metadata Discovery Complete")
        print(f"{'═'*80}")
        print("""
Client now knows:
  ✓ Where to send authorization requests
  ✓ Where to exchange tokens
  ✓ What scopes are available
  ✓ What authentication methods are supported
  ✓ PKCE configuration (S256 required)

Ready to begin OAuth flow!
        """)
    
    async def connect(self):
        """
        Connect to MCP Server
        
        Complete MCP OAuth Connection Flow:
        
        Step 1: Discover OAuth Metadata
        Step 2: Client Registration (PKCE Preparation)
        Step 3: Get Authorization URL
        Step 4: User Authorization (Browser Interaction)
        Step 5: Authorization Code Exchange
        Step 6: Token Verification
        Step 7: MCP Protocol Initialization
        Step 8: Capability Discovery
        """
        print("\n" + "="*80)
        print("🚀 Starting MCP OAuth Connection Flow")
        print("="*80)
        
        # Step 1: Discover OAuth metadata
        if self.auth == "oauth":
            await self._discover_oauth_metadata()
        
        # Steps 2-7: OAuth Authentication Flow (Detailed Breakdown)
        print("\n" + "="*80)
        print("🔐 STEP 2: PKCE Parameter Generation")
        print("="*80)
        print("""
PKCE (Proof Key for Code Exchange) - RFC 7636
Purpose: Prevent authorization code interception attacks in public clients

Process:
1. Generate a random code_verifier (43-128 characters)
2. Calculate code_challenge = BASE64URL(SHA256(code_verifier))
3. Store code_verifier locally (never sent in authorization request)
4. Send code_challenge in authorization request

Implementation:
───────────────────────────────────────────────────────────────
import secrets
import hashlib
import base64

# Generate code_verifier (cryptographically secure random string)
code_verifier = base64.urlsafe_b64encode(
    secrets.token_bytes(32)
).decode('utf-8').rstrip('=')

# Example: "dBjftJeZ412CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk"

# Calculate code_challenge
code_challenge = base64.urlsafe_b64encode(
    hashlib.sha256(code_verifier.encode('utf-8')).digest()
).decode('utf-8').rstrip('=')

# Example: "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
───────────────────────────────────────────────────────────────

Generated PKCE Parameters (simulated):
  code_verifier (stored locally):
    dBjftJeZ412CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk
  
  code_challenge (sent in auth request):
    E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM
  
  code_challenge_method: S256 (SHA-256 hash)

Security Benefits:
  ✓ Even if authorization code is intercepted, attacker cannot use it
  ✓ Only client with original code_verifier can exchange code for token
  ✓ Protects against authorization code injection attacks
        """)
        
        input("\nPress Enter to continue to Step 3...")
        
        print("\n" + "="*80)
        print("🔐 STEP 3: Authorization Request")
        print("="*80)
        print("""
Build Authorization URL and Redirect User to Authorization Server

HTTP Request Format:
───────────────────────────────────────────────────────────────
GET https://login.microsoftonline.com/common/oauth2/v2.0/authorize
    ?response_type=code
    &client_id=12345678-1234-1234-1234-123456789012
    &redirect_uri=http://localhost:8080/callback
    &scope=User.Read%20Files.Read%20email%20openid%20profile
    &state=randomly_generated_state_string_abc123
    &code_challenge=E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM
    &code_challenge_method=S256
    &prompt=select_account
───────────────────────────────────────────────────────────────

Parameter Breakdown:
┌────────────────────┬──────────────────────────────────────────┐
│ Parameter          │ Purpose                                  │
├────────────────────┼──────────────────────────────────────────┤
│ response_type      │ "code" - Request authorization code     │
│ client_id          │ Your app's Azure AD application ID      │
│ redirect_uri       │ Where Azure AD redirects after auth     │
│ scope              │ Permissions requested (space-separated) │
│ state              │ Random string to prevent CSRF attacks   │
│ code_challenge     │ PKCE challenge derived from verifier    │
│ code_challenge_    │ "S256" - SHA-256 hashing method         │
│ method             │                                          │
│ prompt             │ "select_account" - Force account picker │
└────────────────────┴──────────────────────────────────────────┘

Scopes Requested:
  • User.Read      - Read user's basic profile
  • Files.Read     - Read user's OneDrive files
  • email          - Access user's email address
  • openid         - OpenID Connect authentication
  • profile        - Access user's profile information

What Happens Next:
  1. FastMCP opens this URL in your default browser
  2. Browser navigates to Microsoft login page
  3. User sees Microsoft's OAuth consent screen
        """)
        
        input("\nPress Enter to continue to Step 4...")
        
        print("\n" + "="*80)
        print("🔐 STEP 4: User Authorization (Browser Interaction)")
        print("="*80)
        print("""
User Interaction Flow:

1. Browser Opens Microsoft Login Page
   ┌──────────────────────────────────────────────────────┐
   │  Microsoft                                   [X]     │
   │  ════════════════════════════════════════════        │
   │                                                       │
   │  Sign in                                              │
   │                                                       │
   │  ┌─────────────────────────────────────────┐        │
   │  │ Email, phone, or Skype                  │        │
   │  └─────────────────────────────────────────┘        │
   │                                                       │
   │  [ Next ]                                             │
   │                                                       │
   └──────────────────────────────────────────────────────┘

2. User Enters Credentials and Logs In

3. Consent Screen Appears
   ┌──────────────────────────────────────────────────────┐
   │  Microsoft                                   [X]     │
   │  ════════════════════════════════════════════        │
   │                                                       │
   │  Permissions requested                                │
   │                                                       │
   │  Azure OAuth MCP Server wants to:                    │
   │                                                       │
   │  ☑ Read your basic profile (User.Read)              │
   │  ☑ Read your OneDrive files (Files.Read)            │
   │  ☑ Access your email address (email)                │
   │  ☑ Sign you in (openid)                             │
   │  ☑ View your basic profile (profile)                │
   │                                                       │
   │  [ Accept ]  [ Cancel ]                              │
   │                                                       │
   └──────────────────────────────────────────────────────┘

4. After User Clicks "Accept":

HTTP Response from Azure AD:
───────────────────────────────────────────────────────────────
HTTP/1.1 302 Found
Location: http://localhost:8080/callback
          ?code=M.R3_BAY.1234567890abcdefghijklmnopqrstuvwxyz...
          &state=randomly_generated_state_string_abc123
          &session_state=12345678-1234-1234-1234-123456789012
───────────────────────────────────────────────────────────────

Authorization Code Received:
  code: M.R3_BAY.1234567890abcdefghijklmnopqrstuvwxyz...
  
  Properties:
    • Single-use only (can only be exchanged once)
    • Short-lived (typically valid for 10 minutes)
    • Must be exchanged with correct code_verifier (PKCE)
    • Bound to the original client_id and redirect_uri

5. Browser Redirects to Localhost
   FastMCP runs a local HTTP server on localhost:8080/callback
   to capture the authorization code automatically.
        """)
        
        input("\nPress Enter to continue to Step 5...")
        
        print("\n" + "="*80)
        print("🔐 STEP 5: Token Exchange")
        print("="*80)
        print("""
Exchange Authorization Code for Access Token

HTTP Request:
───────────────────────────────────────────────────────────────
POST https://login.microsoftonline.com/common/oauth2/v2.0/token
Host: login.microsoftonline.com
Content-Type: application/x-www-form-urlencoded
Content-Length: 428

Request Body (URL-encoded):
grant_type=authorization_code
&code=M.R3_BAY.1234567890abcdefghijklmnopqrstuvwxyz...
&redirect_uri=http://localhost:8080/callback
&client_id=12345678-1234-1234-1234-123456789012
&code_verifier=dBjftJeZ412CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk
&scope=User.Read Files.Read email openid profile
───────────────────────────────────────────────────────────────

Server-Side Verification Process:
1. Azure AD receives the request
2. Validates the authorization code is valid and not expired
3. Verifies code_verifier matches original code_challenge:
   • Compute: SHA256(code_verifier)
   • Compare: result == stored code_challenge
4. Checks redirect_uri matches the original request
5. Verifies client_id is authorized for these scopes
6. If all checks pass, issues tokens

HTTP Response:
───────────────────────────────────────────────────────────────
HTTP/1.1 200 OK
Content-Type: application/json
Cache-Control: no-store
Pragma: no-cache

{
  "token_type": "Bearer",
  "scope": "User.Read Files.Read email openid profile",
  "expires_in": 3600,
  "ext_expires_in": 3600,
  "access_token": "eyJ0eXAiOiJKV1QiLCJub25jZSI6IkFRQUJBQUFB...",
  "refresh_token": "M.R3_BAY.CfDJ8KZcCxvqV3rC6HJ...",
  "id_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI..."
}
───────────────────────────────────────────────────────────────

Token Breakdown:

1. access_token (JWT - JSON Web Token):
   Header:
   {
     "typ": "JWT",
     "alg": "RS256",
     "kid": "RkZCN0FFRTEw..."
   }
   
   Payload:
   {
     "aud": "00000003-0000-0000-c000-000000000000",
     "iss": "https://sts.windows.net/...",
     "iat": 1701234567,
     "exp": 1701238167,
     "sub": "user-subject-id",
     "upn": "user@example.com",
     "scp": "User.Read Files.Read email openid profile"
   }
   
   • Valid for 1 hour (3600 seconds)
   • Used to authenticate API requests
   • Contains user identity and granted permissions

2. refresh_token:
   • Valid for 90 days (default)
   • Used to obtain new access_token when expired
   • Can be revoked by user or admin

3. id_token:
   • Contains user identity information
   • Used for OpenID Connect authentication
   • Not used for API authorization

Token Storage:
  FastMCP stores tokens securely in memory for the session.
  Tokens are never logged or exposed in plaintext.
        """)
        
        input("\nPress Enter to continue to Step 6...")
        
        print("\n" + "="*80)
        print("🔐 STEP 6: Token Verification")
        print("="*80)
        print("""
Verify Access Token Validity and Extract User Information

Method 1: Local JWT Validation
───────────────────────────────────────────────────────────────
1. Decode JWT header and payload (base64)
2. Verify signature using Microsoft's public keys
3. Check claims:
   • exp (expiration) - Token not expired
   • aud (audience) - Intended for our application
   • iss (issuer) - Issued by Microsoft
   • scp (scope) - Has required permissions

Example Code:
import jwt
from jwt import PyJWKClient

# Get Microsoft's public keys
jwks_url = "https://login.microsoftonline.com/common/discovery/v2.0/keys"
jwks_client = PyJWKClient(jwks_url)

# Get signing key
signing_key = jwks_client.get_signing_key_from_jwt(access_token)

# Verify and decode
decoded = jwt.decode(
    access_token,
    signing_key.key,
    algorithms=["RS256"],
    audience="00000003-0000-0000-c000-000000000000"
)
───────────────────────────────────────────────────────────────

Method 2: Call Microsoft Graph API /me Endpoint (Live Verification)
───────────────────────────────────────────────────────────────
HTTP Request:
GET https://graph.microsoft.com/v1.0/me
Host: graph.microsoft.com
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJub25jZSI6IkFRQUJBQUFB...
Accept: application/json

HTTP Response:
{
  "id": "12345678-1234-1234-1234-123456789012",
  "displayName": "John Doe",
  "mail": "john.doe@example.com",
  "userPrincipalName": "john.doe@example.com",
  "givenName": "John",
  "surname": "Doe",
  "jobTitle": "Software Engineer",
  "officeLocation": "Building 1",
  "mobilePhone": null,
  "businessPhones": ["+1 234 567 8900"]
}
───────────────────────────────────────────────────────────────

Validation Results:
  ✓ Token signature valid (verified with Microsoft public key)
  ✓ Token not expired (exp claim checked)
  ✓ Token audience matches our application
  ✓ Token issued by Microsoft (iss claim verified)
  ✓ Required scopes present in token
  ✓ User identity retrieved: john.doe@example.com

Token is now ready for use in API requests!
        """)
        
        input("\nPress Enter to continue to Step 7...")
        
        print("\n" + "="*80)
        print("🔐 STEP 7: Establish Authenticated Session")
        print("="*80)
        print("""
FastMCP Client Stores Token and Prepares for MCP Protocol Communication

Session Setup:
───────────────────────────────────────────────────────────────
1. Store access_token in client session
   • Token stored in memory (never persisted to disk)
   • Automatically attached to all subsequent MCP requests
   
2. Configure HTTP client with authentication
   • All HTTP requests include Authorization header
   • Format: "Authorization: Bearer <access_token>"
   
3. Set up token refresh mechanism
   • Monitor token expiration (expires_in: 3600 seconds)
   • Automatically refresh using refresh_token before expiry
   • Ensures uninterrupted service

4. Initialize transport layer
   • Create StreamableHttpTransport instance
   • Configure base URL: http://localhost:8000/mcp
   • Enable keep-alive for persistent connections
───────────────────────────────────────────────────────────────

How Authentication Works in Subsequent Requests:

Every MCP Request:
┌─────────────────────────────────────────────────────────────┐
│ POST http://localhost:8000/mcp                              │
│ Content-Type: application/json                              │
│ Authorization: Bearer eyJ0eXAiOiJKV1QiLCJub25jZSI6Ik...    │
│                                                             │
│ {                                                           │
│   "jsonrpc": "2.0",                                        │
│   "id": 1,                                                 │
│   "method": "tools/list",                                  │
│   "params": {}                                             │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ MCP Server (main.py)                                        │
│   1. Extract Bearer token from Authorization header        │
│   2. Validate token (check signature, expiration)          │
│   3. If valid, process MCP request                         │
│   4. Return response                                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ MCP Server Calls Microsoft Graph API                       │
│   • Forwards same access_token to Graph API               │
│   • Graph API validates token independently               │
│   • Returns user data if token valid                      │
└─────────────────────────────────────────────────────────────┘

Security Considerations:
  ✓ Token transmitted over HTTPS only (in production)
  ✓ Token never stored in logs or visible in UI
  ✓ Token automatically expires after 1 hour
  ✓ Refresh token rotates on each use (in production)
  ✓ User can revoke access at any time in Azure AD

Session Ready!
  The OAuth 2.0 + PKCE authentication flow is complete.
  Client can now securely call MCP tools with user's identity.
        """)
        
        # Now FastMCP will execute Steps 2-7 automatically
        print("\n" + "="*80)
        print("🚀 Executing OAuth Flow")
        print("="*80)
        print("""
FastMCP Client will now execute Steps 2-7 automatically.

What will happen:
1. Generate PKCE parameters (code_verifier, code_challenge)
2. Build authorization URL with all parameters
3. Open your default browser → Microsoft login page
4. You interact with browser (login + consent)
5. Capture authorization code from redirect
6. Exchange code for access_token
7. Verify token and establish session

⚠️  IMPORTANT: Browser will open in 3 seconds...
    Please complete the login and consent process.
        """)
        
        import time
        for i in range(3, 0, -1):
            print(f"   Opening browser in {i}...", end='\r')
            time.sleep(1)
        
        print("\n🌐 Opening browser now...")
        
        # Create transport layer
        transport = StreamableHttpTransport(self.server_url)
        
        # Create client (triggers full OAuth flow if auth="oauth")
        self.client = Client(transport, auth=self.auth)
        
        # Establish connection (triggers initialize handshake and OAuth flow)
        await self.client.__aenter__()
        
        print("\n" + "="*80)
        print("✅ OAuth Authentication Successful!")
        print("="*80)
        print("\nAccess token obtained and stored. Ready for MCP protocol communication.\n")
        
        input("Press Enter to continue to Step 8 (MCP Protocol Initialization)...")
        
        # Step 8: MCP Protocol Initialization
        print("\n" + "="*80)
        print("📡 STEP 8: MCP Protocol Initialization")
        print("="*80)
        print("""
MCP (Model Context Protocol) Handshake Process

The MCP protocol uses JSON-RPC 2.0 for all communication.
Before any operations, client and server must exchange capabilities.

═══════════════════════════════════════════════════════════════
Phase 1: Client Sends Initialize Request
═══════════════════════════════════════════════════════════════

HTTP Request:
───────────────────────────────────────────────────────────────
POST http://localhost:8000/mcp
Content-Type: application/json
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJub25jZSI6IkFRQUJBQUFB...
Content-Length: 245

Request Body:
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "roots": {
        "listChanged": true
      },
      "sampling": {}
    },
    "clientInfo": {
      "name": "fastmcp-client",
      "version": "0.1.0"
    }
  }
}
───────────────────────────────────────────────────────────────

Parameters Explained:
┌──────────────────┬───────────────────────────────────────────┐
│ Field            │ Purpose                                   │
├──────────────────┼───────────────────────────────────────────┤
│ protocolVersion  │ MCP protocol version client supports     │
│ capabilities     │ Client's capabilities:                    │
│   - roots        │   Can handle root directory changes      │
│   - sampling     │   Can perform LLM sampling if requested  │
│ clientInfo       │ Client identification and version        │
└──────────────────┴───────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════
Phase 2: Server Responds with Capabilities
═══════════════════════════════════════════════════════════════

HTTP Response:
───────────────────────────────────────────────────────────────
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 312

Response Body:
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
  "capabilities": {
      "tools": {
        "listChanged": true
      },
      "resources": {
        "subscribe": false,
        "listChanged": false
      },
      "prompts": {
        "listChanged": false
      },
      "logging": {}
  },
  "serverInfo": {
    "name": "Azure OAuth MCP Server",
      "version": "1.0.0"
    }
  }
}
───────────────────────────────────────────────────────────────

Server Capabilities Breakdown:
┌──────────────────┬───────────────────────────────────────────┐
│ Capability       │ Description                               │
├──────────────────┼───────────────────────────────────────────┤
│ tools            │ ✓ Server provides executable tools       │
│   listChanged    │   ✓ Server can notify of tool changes    │
├──────────────────┼───────────────────────────────────────────┤
│ resources        │ ✓ Server provides data resources         │
│   subscribe      │   ✗ No resource subscriptions            │
│   listChanged    │   ✗ No resource change notifications     │
├──────────────────┼───────────────────────────────────────────┤
│ prompts          │ ✓ Server provides prompt templates       │
│   listChanged    │   ✗ No prompt change notifications       │
├──────────────────┼───────────────────────────────────────────┤
│ logging          │ ✓ Server supports logging operations     │
└──────────────────┴───────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════
Phase 3: Client Sends Initialized Notification
═══════════════════════════════════════════════════════════════

HTTP Request:
───────────────────────────────────────────────────────────────
POST http://localhost:8000/mcp
Content-Type: application/json
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJub25jZSI6IkFRQUJBQUFB...
Content-Length: 78

Request Body:
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized",
  "params": {}
}
───────────────────────────────────────────────────────────────

Note: This is a notification (no "id" field), so no response expected.

Purpose: Informs server that client has completed initialization
         and is ready to receive requests and notifications.

═══════════════════════════════════════════════════════════════
Handshake Complete!
═══════════════════════════════════════════════════════════════

✓ Protocol version negotiated: 2024-11-05
✓ Client capabilities shared with server
✓ Server capabilities received and understood
✓ Client confirmed ready for operations

The MCP session is now fully established.
Client can now call tools/list, resources/list, tools/call, etc.
        """)
        print("✅ MCP Protocol Handshake Complete")
        
        # Step 9: Discover available tools and resources
        await self._discover_capabilities()
    
    async def _discover_capabilities(self):
        """
        Step 9: Discover Server Capabilities
        
        MCP Capability Discovery:
        - list_tools: Get all available tools
        - list_resources: Get all available resources
        - list_prompts: Get all available prompt templates
        """
        input("\nPress Enter to continue to Step 9 (Capability Discovery)...")
        
        print("\n" + "="*80)
        print("🔍 STEP 9: Discover Server Capabilities")
        print("="*80)
        print("""
After initialization, client must discover what the server offers.
This is done through MCP's list operations.

═══════════════════════════════════════════════════════════════
Operation 1: List Available Tools
═══════════════════════════════════════════════════════════════

Tools are executable functions that the server provides.
They follow the JSON Schema specification for parameters.

HTTP Request:
───────────────────────────────────────────────────────────────
POST http://localhost:8000/mcp
Content-Type: application/json
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJub25jZSI6IkFRQUJBQUFB...
Content-Length: 89

Request Body:
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}
───────────────────────────────────────────────────────────────
        """)
        
        try:
            tools = await self.client.list_tools()
            self.tools_cache = tools
            
            print(f"HTTP Response:")
            print(f"───────────────────────────────────────────────────────────────")
            print(f"HTTP/1.1 200 OK")
            print(f"Content-Type: application/json")
            print(f"Content-Length: [varies]")
            print(f"")
            print(f"Response Body:")
            print(f"{{")
            print(f'  "jsonrpc": "2.0",')
            print(f'  "id": 2,')
            print(f'  "result": {{')
            print(f'    "tools": [')
            
            for i, tool in enumerate(tools, 1):
                comma = "," if i < len(tools) else ""
                print(f'      {{')
                print(f'        "name": "{tool.name}",')
                print(f'        "description": "{tool.description}",')
                
                if hasattr(tool, 'inputSchema'):
                    schema = tool.inputSchema
                    print(f'        "inputSchema": {{')
                    print(f'          "type": "{schema.get("type", "object")}",')
                    
                    if 'properties' in schema:
                        print(f'          "properties": {{')
                        props = schema['properties']
                        prop_list = list(props.items())
                        for j, (prop_name, prop_def) in enumerate(prop_list):
                            prop_comma = "," if j < len(prop_list) - 1 else ""
                            prop_type = prop_def.get('type', 'string')
                            prop_desc = prop_def.get('description', '')
                            print(f'            "{prop_name}": {{')
                            print(f'              "type": "{prop_type}",')
                            print(f'              "description": "{prop_desc}"')
                            print(f'            }}{prop_comma}')
                        print(f'          }},')
                    
                    if 'required' in schema and schema['required']:
                        required_str = '", "'.join(schema['required'])
                        print(f'          "required": ["{required_str}"]')
                    else:
                        print(f'          "required": []')
                    
                    print(f'        }}')
                else:
                    print(f'        "inputSchema": {{"type": "object", "properties": {{}}}}')
                
                print(f'      }}{comma}')
            
            print(f'    ]')
            print(f'  }}')
            print(f'}}')
            print(f"───────────────────────────────────────────────────────────────")
            
            print(f"\n📊 Tools Summary:")
            print(f"   Total Tools: {len(tools)}")
            for i, tool in enumerate(tools, 1):
                print(f"\n   {i}. {tool.name}")
                print(f"      Description: {tool.description}")
                if hasattr(tool, 'inputSchema'):
                    schema = tool.inputSchema
                    if isinstance(schema, dict) and 'properties' in schema:
                        params = list(schema['properties'].keys())
                        required = schema.get('required', [])
                        print(f"      Parameters: {', '.join(params)}")
                        if required:
                            print(f"      Required: {', '.join(required)}")
                        else:
                            print(f"      Required: none")
            
        except Exception as e:
            print(f"  ❌ Failed to get tool list: {e}")
        
        # Get all resources
        print(f"\n{'═'*80}")
        print("Operation 2: List Available Resources")
        print(f"{'═'*80}")
        print("""
Resources are data sources or content that the server exposes.
They can be files, database records, API endpoints, etc.

HTTP Request:
───────────────────────────────────────────────────────────────
POST http://localhost:8000/mcp
Content-Type: application/json
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJub25jZSI6IkFRQUJBQUFB...
Content-Length: 95

Request Body:
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "resources/list",
  "params": {}
}
───────────────────────────────────────────────────────────────
        """)
        
        try:
            resources = await self.client.list_resources()
            self.resources_cache = resources
            
            print(f"HTTP Response:")
            print(f"───────────────────────────────────────────────────────────────")
            print(f"HTTP/1.1 200 OK")
            print(f"Content-Type: application/json")
            print(f"")
            print(f"Response Body:")
            
            if resources:
                print(f"{{")
                print(f'  "jsonrpc": "2.0",')
                print(f'  "id": 3,')
                print(f'  "result": {{')
                print(f'    "resources": [')
                
                for i, resource in enumerate(resources, 1):
                    comma = "," if i < len(resources) else ""
                    print(f'      {{')
                    print(f'        "uri": "{resource.uri if hasattr(resource, "uri") else "N/A"}",')
                    print(f'        "name": "{resource.name}",')
                    print(f'        "description": "{resource.description}",')
                    print(f'        "mimeType": "{resource.mimeType if hasattr(resource, "mimeType") else "application/octet-stream"}"')
                    print(f'      }}{comma}')
                
                print(f'    ]')
                print(f'  }}')
                print(f'}}')
                print(f"───────────────────────────────────────────────────────────────")
                
                print(f"\n📊 Resources Summary:")
                print(f"   Total Resources: {len(resources)}")
                for i, resource in enumerate(resources, 1):
                    print(f"\n   {i}. {resource.name}")
                    print(f"      Description: {resource.description}")
                    if hasattr(resource, 'uri'):
                        print(f"      URI: {resource.uri}")
                    if hasattr(resource, 'mimeType'):
                        print(f"      MIME Type: {resource.mimeType}")
            else:
                print(f"{{")
                print(f'  "jsonrpc": "2.0",')
                print(f'  "id": 3,')
                print(f'  "result": {{')
                print(f'    "resources": []')
                print(f'  }}')
                print(f'}}')
                print(f"───────────────────────────────────────────────────────────────")
                print("\n📊 Resources Summary:")
                print("   Total Resources: 0")
                print("   (Server does not expose any resources)")
                
        except Exception as e:
            print(f"  ⚠️  Failed to get resource list: {e}")
            print(f"     Server may not support resources capability")
        
        # Prompts list (optional)
        print(f"\n{'═'*80}")
        print("Operation 3: List Available Prompts (Optional)")
        print(f"{'═'*80}")
        print("""
Prompts are reusable templates for LLM interactions.
Server may provide pre-configured prompts for common tasks.

HTTP Request:
───────────────────────────────────────────────────────────────
POST http://localhost:8000/mcp
Content-Type: application/json
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJub25jZSI6IkFRQUJBQUFB...
Content-Length: 93

Request Body:
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "prompts/list",
  "params": {}
}
───────────────────────────────────────────────────────────────

HTTP Response:
───────────────────────────────────────────────────────────────
HTTP/1.1 200 OK
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "prompts": []
  }
}
───────────────────────────────────────────────────────────────

Note: This server does not provide pre-configured prompts.
      (This is normal - prompts are optional in MCP)
        """)
        
        print("\n" + "="*80)
        print("✅ Capability Discovery Complete")
        print("="*80)
        print(f"""
Summary of Available Capabilities:
  • Tools: {len(tools)} available
  • Resources: {len(resources) if resources else 0} available  
  • Prompts: 0 available

The client now knows all server capabilities and can call tools.
        """)
    
    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        Convert MCP Tools to LLM Function Calling Format
        
        MCP Tool Description → OpenAI Function Schema
        
        MCP Tool Format:
        {
            "name": "tool_name",
            "description": "Tool description",
            "inputSchema": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }
        
        OpenAI Function Format:
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "description": "Tool description",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }
        """
        llm_tools = []
        
        for tool in self.tools_cache:
            llm_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "No description",
                    "parameters": tool.inputSchema if hasattr(tool, 'inputSchema') else {
                        "type": "object",
                        "properties": {},
                    }
                }
            }
            llm_tools.append(llm_tool)
        
        return llm_tools
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call MCP Tool
        
        MCP Tool Calling Flow:
        1. Client sends call_tool request (with OAuth token)
        2. Server verifies token
        3. Server executes tool logic
        4. Server returns result
        
        Args:
            name: Tool name
            arguments: Tool arguments
        
        Returns:
            Tool execution result
        """
        print("\n" + "="*80)
        print(f"🔧 Calling Tool: {name}")
        print("="*80)
        
        print(f"\n📤 MCP REQUEST")
        print(f"{'─'*80}")
        request_payload = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }
        print(json.dumps(request_payload, indent=2, ensure_ascii=False))
        
        if self.auth == "oauth":
            print(f"\n🔐 Authentication Details:")
            print(f"{'─'*80}")
            print(f"   Authentication Method: OAuth 2.0 Bearer Token")
            print(f"   HTTP Header: Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...")
            print(f"\n   Full HTTP Request:")
            print(f"   POST {self.server_url}")
            print(f"   Content-Type: application/json")
            print(f"   Authorization: Bearer <access_token>")
            print(f"   ")
            print(f"   Body:")
            print(f"   {json.dumps(request_payload, indent=3, ensure_ascii=False)}")
        
        try:
            result = await self.client.call_tool(name, arguments)
            
            print(f"\n📥 MCP RESPONSE")
            print(f"{'─'*80}")
            print(f"   HTTP/1.1 200 OK")
            print(f"   Content-Type: application/json")
            print(f"   ")
            print(f"   Body:")
            response_data = result.data if hasattr(result, 'data') else result
            response_payload = {
                "jsonrpc": "2.0",
                "id": 4,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(response_data, ensure_ascii=False) if isinstance(response_data, (dict, list)) else str(response_data)
                        }
                    ]
                }
            }
            print(f"   {json.dumps(response_payload, indent=3, ensure_ascii=False)}")
            
            print(f"\n{'─'*80}")
            print("✅ Tool Call Successful")
            print("="*80)
            
            return response_data
        except Exception as e:
            error_msg = f"Tool call failed: {str(e)}"
            print(f"\n❌ {error_msg}")
            print("="*80)
            return {"error": error_msg}
    
    async def disconnect(self):
        """Disconnect from server"""
        if self.client:
            await self.client.__aexit__(None, None, None)
            print("\n🔌 Disconnected from MCP Server")


# ============================================================================
# AI Agent - Combining LLM and MCP
# ============================================================================

class MCPAgent:
    """
    MCP AI Agent
    
    This Agent combines DeepSeek LLM and MCP Client to implement:
    1. Receive user queries
    2. LLM understands queries and decides whether to call tools
    3. Call MCP tools to retrieve data
    4. LLM generates final answers based on tool-returned data
    
    This is the core pattern of Agentic AI:
    - Reasoning: LLM understanding and planning
    - Acting: Calling external tools
    - Observing: Getting tool results
    - Responding: Generating final answers
    """
    
    def __init__(self, deepseek_client: DeepSeekClient, mcp_client: MCPClientWrapper):
        """
        Initialize Agent
        
        Args:
            deepseek_client: DeepSeek LLM client
            mcp_client: MCP client wrapper
        """
        self.deepseek = deepseek_client
        self.mcp = mcp_client
        self.conversation_history: List[Dict[str, Any]] = []
        
        # System prompt: Define agent behavior
        self.system_prompt = """You are an intelligent assistant that can help users complete tasks by calling tools.

You have access to the following MCP tools to retrieve information or perform operations.
When a user's request requires information provided by these tools, you should call the appropriate tool.

Remember:
1. Carefully analyze the user's question and determine if tools need to be called
2. If multiple pieces of information are needed, you can call tools multiple times
3. Answer questions based on real data returned by tools
4. If tool calls fail, explain the situation to the user
5. Keep answers concise, accurate, and helpful
"""
    
    async def chat(self, user_message: str) -> str:
        """
        Chat with User
        
        Complete Agentic Workflow:
        
        1. User Input → LLM
        2. LLM Analysis → Decide if tools are needed
        3. If tools are needed:
           a. LLM generates tool call request
           b. Agent calls MCP tools
           c. Get tool results
           d. Return results to LLM
           e. LLM generates answer based on results
        4. If tools are not needed:
           a. LLM directly generates answer
        
        Args:
            user_message: User message
        
        Returns:
            Agent's response
        """
        print(f"\n{'='*70}")
        print(f"💬 Conversation Turn | User Query")
        print(f"{'='*70}")
        print(f"👤 User: {user_message}")
        print(f"{'─'*70}")
        
        # Initialize conversation history (if first conversation)
        if not self.conversation_history:
            self.conversation_history.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        # Add user message
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Get MCP tool list (converted to LLM-usable format)
        tools = self.mcp.get_tools_for_llm()
        
        # Main loop: May require multiple tool call rounds
        max_iterations = 5  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n🧠 LLM Reasoning Engine Started (Round {iteration})")
            print(f"   ├─ Model: DeepSeek Chat")
            print(f"   ├─ Available Tools: {len(tools)}")
            print(f"   └─ Analyzing user intent...")
            
            # Call DeepSeek LLM
            try:
                response = await self.deepseek.chat_completion(
                    messages=self.conversation_history,
                    tools=tools if tools else None,
                )
            except Exception as e:
                error_msg = f"LLM API call failed: {str(e)}"
                print(f"\n❌ {error_msg}")
                return error_msg
            
            # Parse response
            try:
                choice = response["choices"][0]
                message = choice["message"]
                finish_reason = choice["finish_reason"]
            except (KeyError, IndexError) as e:
                error_msg = f"Failed to parse LLM response: {str(e)}"
                print(f"\n❌ {error_msg}")
                print(f"Response: {json.dumps(response, indent=2, ensure_ascii=False)[:500]}")
                return error_msg
            
            # Add LLM response to history
            self.conversation_history.append(message)
            
            # Case 1: LLM decides to call tools
            if finish_reason == "tool_calls" and "tool_calls" in message:
                print(f"\n{'═'*70}")
                print(f"   💡 LLM DECIDES TO CALL TOOLS")
                print(f"{'═'*70}")
                print(f"""
   The LLM has analyzed your question and determined that it needs
   to call external tools to retrieve information.
   
   This demonstrates "Agentic AI" - the LLM is acting as an intelligent
   orchestrator that knows when and how to use available tools.
                """)
                
                # Execute all tool calls
                for idx, tool_call in enumerate(message["tool_calls"], 1):
                    tool_name = tool_call["function"]["name"]
                    tool_call_id = tool_call["id"]
                    
                    print(f"\n{'─'*70}")
                    print(f"   Tool Call #{idx}")
                    print(f"{'─'*70}")
                    
                    # Parse tool arguments with error handling
                    arguments_str = tool_call["function"]["arguments"]
                    try:
                        if isinstance(arguments_str, str):
                            # Handle empty string or whitespace
                            if not arguments_str.strip():
                                tool_args = {}
                            else:
                                tool_args = json.loads(arguments_str)
                        elif isinstance(arguments_str, dict):
                            # Already a dict
                            tool_args = arguments_str
                        else:
                            # Fallback to empty dict
                            tool_args = {}
                    except json.JSONDecodeError as e:
                        print(f"\n   ⚠️  Warning: Failed to parse tool arguments: {e}")
                        print(f"   Raw arguments: {arguments_str}")
                        print(f"   Using empty arguments dict")
                        tool_args = {}
                    
                    print(f"   Selected Tool: {tool_name}")
                    print(f"   Arguments: {json.dumps(tool_args, ensure_ascii=False)}")
                    print(f"   Call ID: {tool_call_id}")
                    
                    print(f"\n   📊 What Happens Next:")
                    print(f"   {'─'*66}")
                    print(f"""
   1. LLM → Agent: "Please call tool '{tool_name}'"
      
   2. Agent → MCP Client: Prepare tool call request
      • Format: JSON-RPC 2.0 protocol
      • Method: tools/call
      • Include OAuth Bearer token in headers
      
   3. MCP Client → MCP Server: Send HTTP POST request
      POST {self.mcp.server_url}
      Authorization: Bearer <access_token>
      Content-Type: application/json
      
      Body:
      {{
        "jsonrpc": "2.0",
        "id": {idx},
        "method": "tools/call",
        "params": {{
          "name": "{tool_name}",
          "arguments": {json.dumps(tool_args, ensure_ascii=False)}
        }}
      }}
      
   {'─'*66}
                    """)
                    
                    print(f"   🚀 Executing tool call...")
                    
                    # Call MCP tool
                    tool_result = await self.mcp.call_tool(tool_name, tool_args)
                    
                    print(f"\n   ✅ Tool execution completed")
                    print(f"   📦 Result summary: {str(tool_result)[:100]}...")
                    if len(str(tool_result)) > 100:
                        print(f"   (Full result passed to LLM for processing)")
                    
                    # Add tool result to conversation history
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": json.dumps(tool_result, ensure_ascii=False)
                    })
                
                # Continue loop to let LLM generate answer based on tool results
                print(f"\n{'═'*70}")
                print(f"   🔄 RETURNING TO LLM FOR SYNTHESIS")
                print(f"{'═'*70}")

                continue
            
            # Case 2: LLM generates final answer
            elif finish_reason == "stop":
                assistant_message = message.get("content", "")
                print(f"\n   ✅ LLM Completed Reasoning, Generated Final Answer")
                print(f"\n{'='*70}")
                print(f"🤖 AI Assistant's Response:")
                print(f"{'='*70}")
                print(f"\n{assistant_message}\n")
                print(f"{'='*70}")
                return assistant_message
            
            # Case 3: Other cases (length limit, etc.)
            else:
                error_msg = f"Unexpected finish_reason: {finish_reason}"
                print(f"\n❌ {error_msg}")
                return error_msg
        
        # Reached maximum iterations
        return "Sorry, I encountered a problem while processing your request (exceeded maximum iterations)."


# ============================================================================
# OAuth Test Function (No LLM Required)
# ============================================================================

async def test_oauth_only():
    """
    Test MCP OAuth Authentication Flow Only, No LLM
    
    Simplified flow focusing on protocol details:
    1. OAuth metadata discovery (detailed)
    2-7. OAuth authentication steps (each detailed)
    8. MCP protocol initialization (detailed)
    9. Capability discovery (detailed tools/list, resources/list)
    
    Note: This version ends after capability discovery.
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║      MCP + OAuth Protocol Deep Dive (No LLM Required)       ║
║                                                              ║
║   Focus: Protocol mechanics, HTTP details, security         ║
║   This version skips LLM integration for faster testing     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

📚 What You'll Learn:
   • OAuth 2.0 + PKCE complete technical breakdown
   • MCP JSON-RPC protocol operations
   • HTTP request/response formats at each step
   • Security mechanisms and token validation

⏱️  Estimated Time: 5-8 minutes
🎯 Focus: Protocol internals, not end-user functionality
    """)
    
    # MCP Server URL - Default to local server
    mcp_server_url = os.getenv(
        "MCP_SERVER_URL",
        "http://localhost:8000/mcp"  # Local test server
    )
    
    print("\n" + "="*80)
    print("📋 Configuration")
    print("="*80)
    print(f"MCP Server URL: {mcp_server_url}")
    
    # Create MCP client
    mcp_client = MCPClientWrapper(
        server_url=mcp_server_url,
        auth="oauth"
    )
    
    try:
        # Connect and authenticate
        await mcp_client.connect()
        
        print("""
╔══════════════════════════════════════════════════════════════╗
║         🎉 Protocol Deep Dive Complete!                     ║
╚══════════════════════════════════════════════════════════════╝

📚 Key Concepts Covered:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. OAuth 2.0 + PKCE Security Flow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ✓ Metadata discovery (RFC 9728, RFC 8414)
   ✓ PKCE parameter generation (code_verifier, code_challenge)
   ✓ Authorization request structure
   ✓ Token exchange with PKCE verification
   ✓ JWT token structure and validation
   ✓ Bearer token authentication

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. MCP Protocol Mechanics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ✓ JSON-RPC 2.0 message format
   ✓ Initialize handshake and capability negotiation
   ✓ Tools/list operation with JSON Schema
   ✓ Resources/list and prompts/list discovery
   ✓ Tool invocation with authenticated requests

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. HTTP-Level Details
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ✓ Complete request/response formats
   ✓ Headers, status codes, content types
   ✓ URL encoding and JSON payloads
   ✓ Authentication header propagation

🔑 Security Chain Verified:

   Browser → Azure AD → Client → MCP Server → Graph API
      ↓         ↓         ↓          ↓            ↓
   Login → Code → Token → Tools → Protected Data

💡 Next Steps:

   • Run full demo with LLM: python mcp_client_with_deepseek.py
   • Review server implementation: main.py
   • Experiment with adding custom tools
   • Deploy to production with proper secret management

Thank you for diving deep into the protocols! 🎓
        """)
        
    finally:
        await mcp_client.disconnect()


# ============================================================================
# Main Program
# ============================================================================

async def main():
    """
    Main Program: Demonstrate Complete MCP + LLM Workflow
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     MCP + OAuth + LLM: Building Intelligent AI Agent        ║
║                                                              ║
║     This example shows how to perfectly combine three       ║
║     major technologies:                                      ║
║     • MCP: Standardized tool calling protocol               ║
║     • OAuth: Secure authentication and authorization        ║
║     • LLM: Intelligent reasoning and decision engine        ║
║                                                              ║
║     Final Result: AI can autonomously decide when to        ║
║     call which tools!                                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

🎯 Complete AI Agent Workflow:

   User Query → LLM Analysis → Decide to Call Tools → OAuth Auth → 
   Call MCP Tools → Get Data → LLM Generate Answer → Return to User

📚 What You'll See:
   • Core working mode of Agentic AI
   • How LLM intelligently selects and calls tools
   • Context management in multi-turn conversations
   • Real-time reasoning process display
    """)
    
    # ========================================================================
    # Step 0: OAuth Authentication Flow Explanation
    # ========================================================================
    
    print("\n" + "="*80)
    print("📚 Complete MCP OAuth Authentication Flow Explanation")
    print("="*80)
    print("""
MCP (Model Context Protocol) + OAuth 2.0 Authentication Flow:

╔══════════════════════════════════════════════════════════════════════════╗
║                       OAuth 2.0 + PKCE Flow Diagram                      ║
╚══════════════════════════════════════════════════════════════════════════╝

                    Client                 MCP Server           Azure AD
                      │                         │                    │
┌─────────────────────┼─────────────────────────┼────────────────────┼──────┐
│ 1. Discovery        │                         │                    │      │
│                     │──── GET /.well-known ──>│                    │      │
│                     │<─── OAuth Metadata ─────│                    │      │
└─────────────────────┼─────────────────────────┼────────────────────┼──────┘
                      │                         │                    │
┌─────────────────────┼─────────────────────────┼────────────────────┼──────┐
│ 2. Registration     │                         │                    │      │
│    (PKCE)           │ Generate:               │                    │      │
│                     │ - code_verifier         │                    │      │
│                     │ - code_challenge        │                    │      │
└─────────────────────┼─────────────────────────┼────────────────────┼──────┘
                      │                         │                    │
┌─────────────────────┼─────────────────────────┼────────────────────┼──────┐
│ 3. Authorization    │                         │                    │      │
│                     │─── GET /authorize ──────>│                    │      │
│                     │    + code_challenge     │                    │      │
│                     │                         │                    │      │
│                     │<─── Redirect to Azure ──│                    │      │
│                     │                         │                    │      │
└─────────────────────┼─────────────────────────┼────────────────────┼──────┘
                      │                         │                    │
┌─────────────────────┼─────────────────────────┼────────────────────┼──────┐
│ 4. User Login       │                         │                    │      │
│    & Consent        │ 🌐 Browser Opens ───────────────────────────>│      │
│                     │                         │   Login & Consent  │      │
│                     │<───────────────────────────── auth_code ──────│      │
└─────────────────────┼─────────────────────────┼────────────────────┼──────┘
                      │                         │                    │
┌─────────────────────┼─────────────────────────┼────────────────────┼──────┐
│ 5. Token Exchange   │                         │                    │      │
│                     │─── POST /token ─────────>│                    │      │
│                     │    + auth_code          │                    │      │
│                     │    + code_verifier      │─── Validate ───────>│      │
│                     │                         │<── OK ──────────────│      │
│                     │<─── access_token ───────│                    │      │
│                     │     refresh_token       │                    │      │
└─────────────────────┼─────────────────────────┼────────────────────┼──────┘
                      │                         │                    │
┌─────────────────────┼─────────────────────────┼────────────────────┼──────┐
│ 6. MCP Operations   │                         │                    │      │
│                     │─── tools/list ──────────>│                    │      │
│                     │    Bearer <token>       │                    │      │
│                     │<─── tools ──────────────│                    │      │
│                     │                         │                    │      │
│                     │─── tools/call ──────────>│                    │      │
│                     │    Bearer <token>       │─── Graph API ──────>│      │
│                     │                         │<── user data ───────│      │
│                     │<─── result ─────────────│                    │      │
└─────────────────────┼─────────────────────────┼────────────────────┼──────┘

Key Concepts:
- PKCE (RFC 7636): Proof Key for Code Exchange, prevents code interception
- OAuth Metadata (RFC 9728): Automatic server capability discovery
- Bearer Token: Access token carried in HTTP Authorization header
    """)
    
    input("\nPress Enter to continue...")
    
    # ========================================================================
    # Step 1: Configuration Check
    # ========================================================================
    
    print("\n" + "="*80)
    print("📋 STEP 1: Check Configuration")
    print("="*80)
    
    # DeepSeek API Key
    deepseek_api_key = "sk-908fa9ea8d1e43c682cde3e6ea76fd98"
    if not deepseek_api_key:
        print("❌ Error: DEEPSEEK_API_KEY environment variable not found")
        print("   Please set: export DEEPSEEK_API_KEY='your_api_key'")
        print("\n💡 Tip: If you only want to test MCP OAuth flow, you can skip DeepSeek")
        print("   You can comment out DeepSeek-related code and test MCP tool calling directly")
        return
    print(f"✅ DeepSeek API Key: {deepseek_api_key[:10]}...")
    
    # MCP Server URL
    # Use current project's server as example
    # MCP Server URL - Default to local server
    mcp_server_url = os.getenv(
        "MCP_SERVER_URL",
        "http://localhost:8000/mcp"  # Local test server
    )
    print(f"✅ MCP Server URL: {mcp_server_url}")
    print(f"\n📍 Server Information:")
    base_url = mcp_server_url.rsplit('/mcp', 1)[0]
    print(f"   Base URL: {base_url}")
    print(f"   MCP Endpoint: {mcp_server_url}")
    print(f"   OAuth Metadata: {base_url}/.well-known/oauth-protected-resource")
    print(f"\n💡 Tip: Make sure local server is running!")
    print(f"   If not started yet, run in another terminal: .venv\\Scripts\\python.exe main.py")
    
    # ========================================================================
    # Step 2: Initialize Clients
    # ========================================================================
    
    print("\n" + "="*80)
    print("📋 STEP 2: Initialize Clients")
    print("="*80)
    
    # Create DeepSeek client
    print("\n🤖 Creating DeepSeek LLM Client...")
    deepseek_client = DeepSeekClient(api_key=deepseek_api_key)
    print("   ✅ DeepSeek client created")
    print(f"   API Base URL: {deepseek_client.base_url}")
    print(f"   Model: deepseek-chat")
    
    # Create MCP client
    print("\n🔌 Creating MCP Client...")
    print(f"   Server URL: {mcp_server_url}")
    print(f"   Authentication Method: OAuth 2.0")
    mcp_client = MCPClientWrapper(
        server_url=mcp_server_url,
        auth="oauth"  # Use OAuth authentication
    )
    print("   ✅ MCP client wrapper created")
    
    print("\n⚠️  Important Notice:")
    print("   The OAuth authentication flow will now execute, which may:")
    print("   1. Open your default browser")
    print("   2. Redirect to Microsoft login page")
    print("   3. Ask you to log in and authorize the app")
    print("   4. Automatically return and continue execution after completion")
    print("\n   If the browser doesn't open automatically, please manually copy the URL to your browser")
    
    input("\nPress Enter to continue and start OAuth authentication flow...")
    
    try:
        # Connect to MCP server
        await mcp_client.connect()
        
        # ====================================================================
        # Step 3: Create AI Agent
        # ====================================================================
        
        print("\n📋 STEP 3: Create AI Agent")
        print("-" * 60)
        
        agent = MCPAgent(
            deepseek_client=deepseek_client,
            mcp_client=mcp_client
        )
        print("✅ AI Agent created")
        
        # ====================================================================
        # Step 10: Use LLM Agent for Intelligent Conversation
        # ====================================================================
        
        input("\nPress Enter to continue to Step 10 (LLM-Powered AI Agent)...")
        
        print("\n" + "="*80)
        print("📋 STEP 10: LLM-Powered AI Agent")
        print("="*80)
        print("""
Now we integrate an LLM (DeepSeek) to create an intelligent agent.

The LLM can:
  • Understand user questions in natural language
  • Decide which tools to call (if any)
  • Call multiple tools in sequence if needed
  • Synthesize tool results into coherent answers

This is "Agentic AI" - LLM acts as a reasoning engine that orchestrates
tool usage to accomplish user goals.

Architecture:
  User Query → LLM Reasoning → Tool Selection → MCP Tool Call
            ← LLM Synthesis ← Tool Result    ← OAuth Auth
        """)
        
        print("\nTips:")
        print("  - Enter your question, Agent will automatically call tools and respond")
        print("  - Enter 'quit' or 'exit' to quit")
        print("  - Enter 'help' to view available tools")
        print()
        
        # Interactive conversation loop
        print("💬 You can now ask questions freely:")
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == "help":
                    print("\nAvailable Tools:")
                    for tool in mcp_client.tools_cache:
                        print(f"  - {tool.name}: {tool.description}")
                    continue
                
                # Chat with Agent
                await agent.chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupt detected, exiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    finally:
        # ====================================================================
        # Cleanup Resources
        # ====================================================================
        
        print("\n📋 Cleaning up resources...")
        print("-" * 60)
        
        await mcp_client.disconnect()
        await deepseek_client.close()
        
        print("✅ Resources cleaned up")
        print("\nThank you for using!")


# ============================================================================
# MCP Protocol Key Concepts Summary
# ============================================================================

"""
📚 MCP Protocol Key Concepts Summary
====================================

1. **Protocol Design Principles**
   - Client-server architecture
   - Transport-agnostic (supports multiple transport protocols)
   - Based on JSON-RPC 2.0
   - Strongly typed Schema (using JSON Schema)

2. **Core Capabilities**
   - Tools: Executable operations provided by the server
   - Resources: Data sources exposed by the server
   - Prompts: Predefined prompt templates
   - Sampling: Server requests client to perform LLM inference

3. **Message Types**
   - Request: Client request, requires response
   - Response: Server response
   - Notification: One-way notification, no response needed

4. **Lifecycle**
   - Initialize: Establish connection, negotiate capabilities
   - Operations: Normal operations (list/call/read, etc.)
   - Shutdown: Graceful shutdown

5. **Security**
   - OAuth 2.0 support
   - API Key support
   - Custom authentication support
   - HTTPS transport encryption

6. **Best Practices**
   - Use typed Schema definitions
   - Implement error handling and retry logic
   - Support operation cancellation
   - Provide clear tool descriptions
   - Implement logging and monitoring

7. **Relationship with Function Calling**
   - MCP Tools ≈ OpenAI Function Calling
   - MCP provides standardized tool definition format
   - MCP supports cross-platform, cross-LLM usage
   - MCP adds additional concepts like resources and prompts

8. **Application Scenarios**
   - RAG (Retrieval Augmented Generation)
   - Database queries
   - API calls
   - File operations
   - Workflow automation
   - Multi-modal processing

More Information:
- MCP Specification: https://modelcontextprotocol.io/specification/2025-06-18
- FastMCP Documentation: https://gofastmcp.com
- DeepSeek API: https://platform.deepseek.com/api-docs/
"""


if __name__ == "__main__":
    """
    Complete OAuth 2.0 + PKCE + MCP Protocol Deep Dive
    
    This script provides an in-depth, step-by-step walkthrough:
    
    PART 1: OAuth 2.0 + PKCE Flow (Steps 1-7)
      1. Metadata Discovery (RFC 9728, RFC 8414)
      2. PKCE Parameter Generation
      3. Authorization Request
      4. User Authorization (Browser)
      5. Token Exchange
      6. Token Verification
      7. Authenticated Session Establishment
    
    PART 2: MCP Protocol Operations (Steps 8-9)
      8. MCP Initialize Handshake
      9. Capability Discovery (tools/list, resources/list, prompts/list)
    
    PART 3: LLM Integration (Step 10)
      10. Intelligent AI Agent with Tool Selection
    
    Every step shows:
      • Complete HTTP request format (method, headers, body)
      • Complete HTTP response format (status, headers, body)
      • Parameter explanations and security implications
      • Protocol specifications and standards (RFCs)
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        OAuth 2.0 + MCP Protocol: Professional Deep Dive     ║
║                                                              ║
║   Complete technical walkthrough with HTTP-level details    ║
║   Designed for developers who want to understand the        ║
║   internals of OAuth and MCP protocols                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

📋 What This Demo Covers:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1: OAuth 2.0 Security & Authentication
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ RFC 9728 - OAuth 2.0 Protected Resource Metadata
  ✓ RFC 8414 - Authorization Server Metadata Discovery
  ✓ RFC 7636 - PKCE (Proof Key for Code Exchange)
  ✓ Complete Authorization Code Flow breakdown
  ✓ Token structure and JWT validation
  ✓ Bearer token authentication mechanism

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2: MCP Protocol Internals
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ JSON-RPC 2.0 protocol format
  ✓ Initialize handshake and capability negotiation
  ✓ Detailed tools/list response with JSON Schema
  ✓ Resources and prompts discovery

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3: LLM Integration & Agentic AI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ LLM-powered tool selection
  ✓ Reasoning → Acting → Observing pattern
  ✓ Multi-turn conversation with tool calls
  ✓ Interactive Q&A with intelligent tool invocation

📋 Prerequisites:
  • MCP Server running: python main.py
  • Microsoft account for OAuth login
  • DeepSeek API Key: export DEEPSEEK_API_KEY='...'

⏱️  Duration: 10-15 minutes
🎯 Audience: Developers, Security Engineers, Technical Architects

💡 Focus: Protocol details, HTTP formats, security mechanisms
   (Less focus on individual tool demonstrations)
    """)
    
    input("\nPress Enter to start the complete demonstration...")
    
    try:
        print("\n" + "🚀"*40)
        print("Starting Complete OAuth MCP Flow Demonstration")
        print("🚀"*40 + "\n")
        asyncio.run(main())
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

