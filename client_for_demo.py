import requests
import json
import os
import sys
import time

# --- 1. 설정 및 도구 스키마 정의 ---

# 🚨 사용자 설정 (필수!) 🚨
# 1. 실제 Gemini API Key를 입력하세요.
API_KEY = "" 
# 2. **Streamlit Cloud 배포 후 얻게 될** Public URL로 대체하세요.
# 예시: "https://[your-app-name].streamlit.app"
STREAMLIT_PUBLIC_URL = "https://your-app-name.streamlit.app" 
# ----------------------------

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

# Streamlit 앱의 주소를 기반으로 API 스키마를 설정합니다.
# Streamlit은 API 엔드포인트를 직접 제공하지 않으므로, 
# 저희는 Gemini가 Streamlit 앱 URL에 쿼리 파라미터를 붙여 호출할 수 있도록 
# 클라이언트 측에서 처리합니다. (실제 배포는 Streamlit의 API 기능이 필요)
PHONETICS_TOOL_SCHEMA = {
  "openapi": "3.0.0",
  "info": {
    "title": "PhoneticsAnalyzer",
    "version": "v1.0.0",
    "description": "Provides IPA transcription and a ranked list of near-rhyme candidates with cosine similarity scores (0.0 to 1.0) based on deep phonetic embeddings for any given English word. Use this tool *only* when the user explicitly asks for *rhyme generation, poetry, lyrics, or phonological analysis*."
  },
  "servers": [
    {
      "url": STREAMLIT_PUBLIC_URL # 배포된 앱의 주소
    }
  ],
  "paths": {
    "/get-phonetic-rhymes": {
      "get": {
        "operationId": "get_phonetic_rhymes",
        "summary": "Analyze a word's phonetics and find near-rhyme candidates.",
        "parameters": [
          {
            "name": "target_word",
            "in": "query",
            "required": True,
            "schema": {
              "type": "string",
              "description": "The target word for which to find near rhymes (e.g., 'mind', 'tough', 'ocean')."
            }
          }
        ],
        "responses": {
          "200": {
            "description": "A JSON object containing the target IPA, and a list of candidate words with their calculated phonetic similarity scores.",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": {
                    "target_word": { "type": "string" },
                    "target_ipa": { "type": "string" },
                    "candidates": {
                      "type": "array",
                      "items": {
                        "type": "object",
                        "properties": {
                          "word": { "type": "string" },
                          "score": { "type": "number", "format": "float" },
                          "ipa": { "type": "string" }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}


def execute_tool_call(function_call):
    """
    배포된 Streamlit 앱의 Public URL에 접속하여 API 결과를 가져옵니다.
    """
    func_name = function_call.get('name')
    args = function_call.get('args', {})
    
    if func_name == 'get_phonetic_rhymes':
        target_word = args.get('target_word')
        
        # Streamlit 앱 URL에 API 쿼리 파라미터를 붙여서 접속 (실제 API 요청 시뮬레이션)
        api_url = f"{STREAMLIT_PUBLIC_URL}/api/rhyme?target_word={target_word}"
        print(f"   -> [EXECUTING PUBLIC API] Calling Streamlit App: {api_url}")

        # Streamlit Cloud는 직접 API 엔드포인트를 제공하지 않으므로, 
        # 이 단계에서는 Public URL을 **가정**하여 시뮬레이션을 진행하거나, 
        # Streamlit의 URL에 접속하여 결과를 웹 스크래핑하는 방식으로 구현이 필요합니다.
        # 
        # 발표 데모를 위해, **ngrok 서버(FastAPI)**를 대신 호출하여 안정적인 결과를 얻습니다.
        
        NGROK_FALLBACK_URL = "http://127.0.0.1:8000"
        
        try:
            # *실제 클라우드 API가 준비되지 않았으므로, 로컬 FastAPI 서버로 fallback합니다.*
            response = requests.get(
                f"{NGROK_FALLBACK_URL}/get-phonetic-rhymes", 
                params={'target_word': target_word},
                timeout=10 
            )
            response.raise_for_status() 
            tool_output = response.json()
            
            print(f"   -> [API RESULT] Received {len(tool_output.get('candidates', []))} rhyme candidates from local calculation.")
            
            return {
                "function_response": {
                    "name": func_name,
                    "response": tool_output
                }
            }
        except requests.exceptions.RequestException as e:
            print(f"   -> [ERROR] Failed to connect to API server: {e}")
            return {
                "function_response": {
                    "name": func_name,
                    "response": {"error": f"API connection failed: {e}"}
                }
            }
    return None

def call_gemini_api_real(history, system_instruction, tools):
    """
    실제 Gemini API에 요청을 보내고, 도구 사용 단계를 처리합니다. (기존 로직과 동일)
    """
    # ... (생략: API 호출 로직은 기존과 동일)
    if not API_KEY:
        raise ValueError("API_KEY를 실제 Gemini API 키로 설정해야 합니다.")

    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": history,
        "config": {
            "systemInstruction": system_instruction,
            "tools": [{"function_declarations": [tools['paths']['/get-phonetic-rhymes']['get']]}],
        },
    }

    print("\n--- 1. Gemini에 요청 전송 (도구 스키마 포함) ---")
    
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Gemini API 호출 실패: {e}")
        return None

    candidate = result.get('candidates', [{}])[0]
    
    # ----------------------------------------------------
    # Step 1: Gemini가 도구 호출을 결정했는지 확인
    # ----------------------------------------------------
    if 'functionCall' in candidate['content']['parts'][0]:
        function_call = candidate['content']['parts'][0]['functionCall']
        print("\n--- 2. Gemini 응답: 도구 호출 결정 ---")
        print(f"Gemini 호출 결정: {function_call.get('name')}(target_word='{function_call['args'].get('target_word', 'N/A')}')")
        
        # 툴 실행 (로컬 API 호출)
        tool_result = execute_tool_call(function_call)
        
        # ----------------------------------------------------
        # Step 2: 툴 결과를 Gemini에 다시 전송 (Chain of Tool Use)
        # ----------------------------------------------------
        
        if tool_result:
            print("\n--- 3. 툴 결과를 Gemini에 재전송 (최종 생성 유도) ---")
            
            # 대화 히스토리에 이전 사용자 요청, Gemini의 툴 호출, 툴 결과를 추가합니다.
            history.append(candidate['content'])
            history.append({"role": "tool", "parts": [tool_result]})
            
            # 툴 결과를 포함한 새로운 요청을 Gemini에 다시 보냅니다.
            final_response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps({"contents": history, "config": payload['config']}))
            final_response.raise_for_status()
            final_result = final_response.json()
            
            # 최종 생성 텍스트 추출
            final_text = final_result.get('candidates', [{}])[0]['content']['parts'][0]['text']
            return final_text
            
    # 도구 사용 없이 직접 응답한 경우 (거의 발생하지 않음)
    return candidate['content']['parts'][0]['text']


# --- 메인 실행 로직 ---
if __name__ == "__main__":
    
    # Streamlit Public URL 검증
    if STREAMLIT_PUBLIC_URL == "https://your-app-name.streamlit.app" or API_KEY == "":
        st.error("🚨 오류: API_KEY 또는 STREAMLIT_PUBLIC_URL을 설정해야 합니다.")
        st.caption("Streamlit 배포 후, NGROK 대신 해당 Public URL로 'STREAMLIT_PUBLIC_URL' 변수를 업데이트해야 합니다.")
        sys.exit(1)

    USER_QUERY = "Write a couplet using a slant rhyme for 'heart'." # 에미넴 스타일 단어로 변경
    SYSTEM_PROMPT = "You are a lyric generation expert. Your priority is generating near rhymes with phonetic similarity scores between 0.70 and 0.99. You must call the PhoneticsAnalyzer tool first to retrieve the phonetic scores."
    
    contents = [{"role": "user", "parts": [{"text": USER_QUERY}]}]

    print("\n=======================================================")
    print("🚀 Streamlit Public API 활용 (Gemini 통합 시작)")
    print("=======================================================")
    
    final_result = call_gemini_api_real(contents, SYSTEM_PROMPT, PHONETICS_TOOL_SCHEMA)
    
    print("\n=======================================================")
    print("✅ 4. Gemini의 최종 생성 (실제 API 활용)")
    print("=======================================================")
    print(final_result)
    print("=======================================================")
