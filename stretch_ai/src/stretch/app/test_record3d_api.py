#!/usr/bin/env python3
"""
Test Record3D API endpoints to understand the actual interface.
"""

import requests
import json
import sys

def test_all_endpoints(ip: str, port: int = 80):
    """Test all possible Record3D endpoints."""
    base_url = f"http://{ip}:{port}"
    
    # Common endpoints to test
    endpoints = [
        "/",
        "/metadata",
        "/getOffer", 
        "/sendAnswer",
        "/api/metadata",
        "/api/getOffer",
        "/api/sendAnswer",
        "/webrtc/offer",
        "/webrtc/answer",
        "/stream",
        "/status",
        "/info",
        "/config",
    ]
    
    print(f"Testing Record3D API endpoints at {base_url}")
    print("=" * 60)
    
    for endpoint in endpoints:
        try:
            print(f"Testing {endpoint}...")
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            
            print(f"  Status: {response.status_code}")
            print(f"  Headers: {dict(response.headers)}")
            
            # Try to parse as JSON
            try:
                data = response.json()
                print(f"  JSON Data: {json.dumps(data, indent=2)}")
            except:
                # Show first 200 chars of content
                content = response.text[:200]
                if content:
                    print(f"  Content: {content}...")
                    
        except Exception as e:
            print(f"  Error: {e}")
            
        print("-" * 40)
        
    # Test POST to sendAnswer (if getOffer worked)
    print("\nTesting POST endpoints...")
    
    post_endpoints = [
        "/answer",
        "/sendAnswer",
        "/api/sendAnswer", 
        "/webrtc/answer"
    ]
    
    for endpoint in post_endpoints:
        try:
            print(f"Testing POST {endpoint}...")
            
            # Test with empty JSON
            response = requests.post(
                f"{base_url}{endpoint}", 
                json={"test": "data"}, 
                timeout=5
            )
            
            print(f"  Status: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            
        except Exception as e:
            print(f"  Error: {e}")
            
        print("-" * 20)


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_record3d_api.py <device_ip> [port]")
        sys.exit(1)
        
    ip = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    
    test_all_endpoints(ip, port)


if __name__ == "__main__":
    main()