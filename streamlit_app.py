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
# 2. 핵심 계산 함수 (강세 무시, 종성 3개 음소 일치 로직)
# =========================================================

def get_phonetic_tail(word):
    """
    강세와 상관없이 CMUDict에서 단어의 끝 음소(clean ARPAbet)를 추출합니다.
    음소 길이가 3개 미만인 경우 모두 사용합니다.
    """
    word = word.lower()
    if word not in p_dict:
        return None, None
        
    pron_raw = p_dict[word][0] 
    
    # ARPAbet에서 스트레스 마크 제거 (0, 1, 2)
    pron_clean_full = [phon.rstrip('0123') for phon in pron_raw]
    
    # 길이가 3개 미만이면 전체를 사용하고, 3개 이상이면 끝 3개 음소를 라임 유닛으로 사용
    rhyme_unit_clean = pron_clean_full[-3:] if len(pron_clean_full) >= 3 else pron_clean_full
    
    # 원본 pron, 클린 라임 유닛을 반환
    return pron_raw, rhyme_unit_clean

# ---------------------------------------------------------
# 🚨 핵심 수정 부분: ARPAbet을 IPA로 직접 매핑하는 안전 로직
# ---------------------------------------------------------
# eng-to-ipa 라이브러리의 의존성을 낮추고 변환 오류를 줄입니다.
# CMUDict에서 가장 흔하게 발생하는 ARPAbet 기호에 대한 매핑입니다.
ARPABET_TO_IPA_MAP = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ', 
    'B': 'b', 'CH': 'ʧ', 'D': 'd', 'DH': 'ð', 'EH': 'ɛ', 'ER': 'əɹ', 
    'EY': 'eɪ', 'F': 'f', 'G': 'g', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 
    'JH': 'ʤ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 
    'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'ʃ', 
    'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v', 'W': 'w', 
    'Y': 'j', 'Z': 'z', 'ZH': 'ʒ', 'T': 't', 'D': 'd'
}

def arpabet_to_ipa(arpabet_phons):
    """ARPAbet 음소열을 직접 매핑하여 IPA 문자열로 변환합니다."""
    if not arpabet_phons:
        return None
    
    ipa_phons = [ARPABET_TO_IPA_MAP.get(phon.upper(), '') for phon in arpabet_phons]
    
    # 매핑되지 않은 음소(빈 문자열)는 제외하고 문자열로 합칩니다.
    ipa_str = "".join([p for p in ipa_phons if p])
    
    return ipa_str if ipa_str else None
# ---------------------------------------------------------


def calculate_rhyme_score(ipa1, ipa2):
    """두 IPA 문자열의 벡터 유사도 점수 (코사인 유사도)를 계산합니다."""
    
    # 마지막 3개 음소 로직 유지 (근사 라임 기준)
    phons1 = list(ipa1)[-3:]
    phons2 = list(ipa2)[-3:]
    
    # IPA가 3개 미만인 단어도 처리하기 위해 길이를 확인
    if not phons1 or not phons2 or len(phons1) != len(phons2):
        # 짧은 단어끼리는 길이가 같아야 유사도를 측정합니다.
        if len(phons1) < 3 and len(phons1) == len(phons2):
             pass
        else:
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
    """CMUDict 전체를 검색하여 끝 음소가 일치하는 후보를 찾고 점수를 매깁니다."""
    
    # 강세가 없는 단순 종성 추출
    target_pron_raw, target_rhyme_unit = get_phonetic_tail(target_word)
    
    if not target_rhyme_unit:
        return {"target_word": target_word, "target_ipa": "N/A", "raw_arpabet": target_pron_raw, "candidates": []}

    # 라임 유닛 IPA 변환 
    target_ipa = arpabet_to_ipa(target_rhyme_unit)
    
    # IPA 변환이 실패하면 여기서 바로 종료 (IPA가 None이 아님)
    if not target_ipa:
        return {"target_word": target_word, "target_ipa": "N/A", "raw_arpabet": target_pron_raw, "candidates": []}

    candidates_list = []
    
    # ----------------------------------------------------------------
    # CMUDict 전체 스캔 로직
    # ----------------------------------------------------------------
    for word, pron_list in p_dict.items():
        pron_raw = pron_list[0]
        
        # 후보 단어의 끝 음소 추출
        candidate_pron_clean_full = [p.rstrip('0123') for p in pron_raw]
        candidate_rhyme_unit = candidate_pron_clean_full[-len(target_rhyme_unit):]
        
        # 1. 단어 필터링 (자기 자신, 너무 짧은 단어, 라임 유닛 길이 불일치)
        if word == target_word.lower() or len(word) <= 2 or len(candidate_rhyme_unit) != len(target_rhyme_unit):
            continue
            
        # 2. 끝 음소 일치 확인 (가장 단순한 라임 조건)
        if candidate_rhyme_unit == target_rhyme_unit:
            
            # IPA 변환 (점수 계산용)
            candidate_ipa = arpabet_to_ipa(candidate_rhyme_unit) # 라임 유닛만 변환
            
            if candidate_ipa:
                # IPA 기반 음소 임베딩 점수 계산
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
        "raw_arpabet": target_pron_raw, 
        "candidates": candidates_list[:top_n]
    }


# =========================================================
# 3. Streamlit UI (CMUDict가 활성화된 UI)
# =========================================================

st.set_page_config(page_title="Phonetics Analyzer (Simplified Rhyme)", layout="centered")

st.title("🎤 CMUDict 통합: 음소 임베딩 단순 라임 분석")
st.caption("✅ 강세(Stresses)를 무시하고 단어 끝 3개 음소 기준으로 검색합니다.")

st.markdown("""
이 툴은 **CMUDict (13만 단어)**를 활용하여, 
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
        st.warning(f"CMUDict에서 '{input_word}'에 대한 라임 유닛을 찾지 못했습니다. (단어가 사전에 없거나 음소가 3개 미만이거나 너무 흔한 기능어일 수 있습니다.)")

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
