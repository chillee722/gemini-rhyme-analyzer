import streamlit as st
import eng_to_ipa as ipa
import numpy as np
from scipy.spatial.distance import cosine
import json
import nltk
from nltk.corpus import cmudict
import sys
import os

# --- CMUDict 다운로드 및 로드 ---
# Streamlit Cloud 환경에서 NLTK 데이터 다운로드가 필수적이므로, 예외 처리를 일반화합니다.
@st.cache_resource(show_spinner="CMUDict 사전을 로드 중...")
def load_cmudict():
    try:
        # CMUDict 데이터가 로컬에 있는지 확인합니다.
        nltk.data.find('corpora/cmudict')
    except LookupError: # NLTK 데이터가 없을 때 발생하는 일반적인 예외
        # 데이터 다운로드 및 로드
        nltk.download('cmudict')
        
    return cmudict.dict()

p_dict = load_cmudict()

# =========================================================
# 1. 음소 임베딩 정의 (유사도 계산의 핵심 데이터)
# =========================================================

# (예시) 근사 라임 판단에 중요한 음소들의 임베딩 (5차원)
PHONEME_EMBEDDINGS = {
    'æ': np.array([1.0, 0.0, 0.5, 0.0, 0.0]),  # as in 'cat', 'hat'
    'ʌ': np.array([0.9, 0.1, 0.6, 0.0, 0.0]),  # as in 'cut'
    'aɪ': np.array([0.5, 0.8, 0.1, 0.0, 0.0]), # as in 'mind'
    'ɛ': np.array([0.4, 0.7, 0.2, 0.0, 0.0]),  # as in 'spend'
    'i': np.array([0.2, 0.9, 0.1, 0.0, 0.0]),  # as in 'feel'
    't': np.array([0.0, 1.0, 0.0, 1.0, 0.0]),  # Consonant 't'
    'd': np.array([0.0, 0.9, 0.0, 1.0, 0.1]),  # Consonant 'd' (t와 유사)
    'n': np.array([0.0, 0.8, 0.1, 1.0, 0.0]),  # Consonant 'n'
    'nd': np.array([0.0, 0.7, 0.1, 1.0, 0.1]), # Consonant cluster 'nd'
    'r': np.array([0.1, 0.0, 0.8, 0.5, 0.5]),
    'ʊ': np.array([0.3, 0.6, 0.4, 0.0, 0.0]),  # as in 'good'
    'k': np.array([0.0, 0.5, 0.0, 0.5, 0.0]),  # Consonant 'k'
    'v': np.array([0.0, 0.8, 0.0, 1.0, 0.2]),  # Consonant 'v'
    'l': np.array([0.0, 0.0, 0.7, 0.4, 0.0]),  # Consonant 'l'
}

# =========================================================
# 2. 핵심 계산 함수 (CMUDict 활용 로직)
# =========================================================

@st.cache_data(show_spinner=False)
def get_arpabet_and_rhyme_unit(word):
    """CMUDict에서 단어의 ARPAbet 발음과 라임 유닛을 추출합니다."""
    word = word.lower()
    if word not in p_dict:
        # 단어가 CMUDict에 없는 경우
        return None, None, None 

    # CMUDict는 다중 발음을 가질 수 있지만, 첫 번째 발음만 사용합니다.
    pron = p_dict[word][0] 
    
    rhyme_start_index = -1
    
    # 1. 주 강세(1)를 먼저 찾습니다. (가장 중요)
    for i, phon in enumerate(pron):
        if phon.endswith('1'): 
            rhyme_start_index = i
            break
            
    # 2. 주 강세가 없으면 부 강세(2)를 찾습니다. (다중 음절 단어 보조)
    if rhyme_start_index == -1:
        for i, phon in enumerate(pron):
            if phon.endswith('2'):
                rhyme_start_index = i
                break
            
    # 3. 강세 모음이 전혀 없는 단어(함께, to, a 등)는 실패 처리합니다.
    if rhyme_start_index == -1:
        return pron, None, None # 원본 pron은 반환하여 UI에서 디버깅 가능하도록 함.

    # ARPAbet 발음에서 스트레스 마크 제거
    clean_arpabet = [phon.rstrip('0123') for phon in pron]
    # 라임 유닛 추출 (강세 모음부터 끝까지)
    rhyme_unit = clean_arpabet[rhyme_start_index:]
    
    # 원본 pron, 클린 버전, 라임 유닛을 모두 반환
    return pron, clean_arpabet, rhyme_unit

def arpabet_to_ipa(arpabet_phons):
    """ARPAbet 음소열을 eng-to-ipa를 사용하여 IPA 문자열로 변환합니다."""
    arpabet_str = ', '.join(arpabet_phons)
    try:
        # eng-to-ipa 라이브러리의 모드에 주의하여 IPA 문자열을 반환합니다.
        # 공백과 강세 마크를 제거하여 깔끔한 음소열만 남깁니다.
        ipa_str = ipa.convert(arpabet_str, mode='arpabet').strip().replace(' ', '').replace('ˈ', '').replace('ˌ', '')
        return ipa_str
    except Exception:
        return None

def calculate_rhyme_score(ipa1, ipa2):
    """두 IPA 문자열의 벡터 유사도 점수 (코사인 유사도)를 계산합니다."""
    
    phons1 = list(ipa1)[-3:]
    phons2 = list(ipa2)[-3:]
    
    if not phons1 or not phons2 or len(phons1) != len(phons2):
        return 0.0
    
    vec1_list = [PHONEME_EMBEDDINGS.get(p, np.zeros(5)) for p in phons1]
    vec2_list = [PHONEME_EMBEDDINGS.get(p, np.zeros(5)) for p in phons2]
    
    vec1 = np.concatenate(vec1_list)
    vec2 = np.concatenate(vec2_list)
    
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0

    similarity = 1 - cosine(vec1, vec2)
    return max(0, similarity)


@st.cache_data(show_spinner=False)
def get_rhyme_candidates_with_score(target_word: str, top_n=100):
    """CMUDict 전체를 검색하여 라임 유닛이 일치하는 후보를 찾고 점수를 매깁니다."""
    
    # 반환 값이 세 개로 변경됨
    target_pron_raw, target_arpabet_clean, target_rhyme_unit = get_arpabet_and_rhyme_unit(target_word)
    
    if not target_rhyme_unit:
        # 라임 유닛 추출에 실패하면 원본 ARPAbet만 반환하여 UI에 표시 (디버깅용)
        return {"target_word": target_word, "target_ipa": "N/A", "raw_arpabet": target_pron_raw, "candidates": []}

    target_ipa = arpabet_to_ipa(target_rhyme_unit)
    
    candidates_list = []
    
    # ----------------------------------------------------------------
    # CMUDict 전체 스캔 로직
    # ----------------------------------------------------------------
    for word, pron_list in p_dict.items():
        pron_arpabet = pron_list[0]
        pron_clean = [p.rstrip('0123') for p in pron_arpabet]
        
        if word == target_word.lower() or len(word) <= 2 or len(pron_clean) < len(target_rhyme_unit): 
            continue
            
        # 끝 부분이 라임 유닛과 일치하는지 확인 (Near Rhyme 후보군 필터링)
        if pron_clean[-len(target_rhyme_unit):] == target_rhyme_unit:
            
            candidate_ipa = arpabet_to_ipa(pron_clean)
            if candidate_ipa:
                score = calculate_rhyme_score(target_ipa, candidate_ipa) 
                
                candidates_list.append({
                    "word": word,
                    "score": round(score, 2),
                    "ipa": candidate_ipa
                })

    candidates_list.sort(key=lambda x: x['score'], reverse=True)

    return {
        "target_word": target_word,
        "target_ipa": target_ipa,
        "raw_arpabet": target_pron_raw, # 디버깅용으로 추가
        "candidates": candidates_list[:top_n]
    }


# =========================================================
# 3. Streamlit UI (CMUDict가 활성화된 UI)
# =========================================================

st.set_page_config(page_title="Phonetics Analyzer (CMUDict Integrated)", layout="centered")

st.title("🎤 CMUDict 통합: 음소 임베딩 근사 라임 분석")
st.caption("✅ CMUDict 기반으로 전체 영어 단어 검색 가능 (기말 프로젝트 개선 사항 반영)")

st.markdown("""
이 툴은 **수동 딕셔너리** 대신 **CMUDict (13만 단어)**를 활용하여, 
입력 단어와 **음소 유사성**이 높은 모든 단어를 검색하고 점수를 매깁니다. 
이 JSON 결과가 Gemini에게 제공할 **API 응답**입니다.
""")

# 사용자 입력
input_word = st.text_input("분석할 단어를 입력하세요 (예: tough, mind, heart)", "mind")

if input_word:
    st.subheader(f"🔍 '{input_word}'에 대한 CMUDict 기반 분석 결과")
    
    # 계산 로직 실행
    with st.spinner('CMUDict를 스캔하고 음소 임베딩을 계산 중...'):
        analysis_result = get_rhyme_candidates_with_score(input_word)
    
    # --- 디버깅 정보 출력 ---
    st.markdown("#### 🚨 디버깅 정보 (발표 시 숨김 권장)")
    st.markdown(f"**CMUDict 원본 발음 (ARPAbet):** `{analysis_result.get('raw_arpabet')}`")
    st.markdown(f"**대상 단어 IPA (라임 유닛):** `{analysis_result['target_ipa']}`")
    
    # 2. 유사도 테이블 표시
    st.markdown("---")
    st.markdown("#### CMUDict 기반 검색 결과 (상위 100개 중 점수 순 정렬)")
    
    if analysis_result['candidates']:
        # 테이블 데이터 준비
        data = []
        for c in analysis_result['candidates']:
            rhyme_type = "Perfect Rhyme" if c['score'] >= 0.99 else ("Near Rhyme" if c['score'] >= 0.70 else "Slant/Poor Match")
            data.append({
                "Word": c['word'],
                "IPA (Phonetics)": c['ipa'],
                "Phonetic Score": f"{c['score']:.2f}",
                "Rhyme Type": rhyme_type
            })
        
        st.dataframe(data, use_container_width=True, hide_index=True)
    else:
        st.warning(f"CMUDict에서 '{input_word}'에 대한 라임 유닛을 찾지 못했습니다. (단어가 사전에 없거나 너무 짧을 수 있습니다.)")
        if analysis_result.get('raw_arpabet'):
            st.error("오류 원인: **주 강세(1) 또는 부 강세(2)**를 찾지 못하여 라임 유닛 추출에 실패했습니다. 단어의 강세를 확인해주세요.")

    # 3. Gemini가 받을 API 응답 (발표 강조점)
    st.markdown("---")
    st.markdown("#### 🤖 Gemini에게 제공할 최종 API 응답 (JSON)")
    # UI에 표시되는 JSON에서는 디버깅 정보 제외
    final_json_output = {
        "target_word": analysis_result["target_word"],
        "target_ipa": analysis_result["target_ipa"],
        "candidates": analysis_result["candidates"]
    }
    st.code(json.dumps(final_json_output, indent=2), language='json')
