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
@st.cache_resource(show_spinner="CMUDict 사전을 로드 중...")
def load_cmudict():
    try:
        nltk.data.find('corpora/cmudict')
    except LookupError:
        nltk.download('cmudict')
        
    return cmudict.dict()

p_dict = load_cmudict()

# =========================================================
# 1. 음소 임베딩 정의 (ARPAbet 기호로 통일)
# =========================================================

# PHONEME_EMBEDDINGS의 키를 CMUDict의 ARPAbet 기호로 변경하여
# IPA 변환 없이 ARPAbet 자체를 벡터화할 수 있도록 통일시킵니다.
PHONEME_EMBEDDINGS = {
    # 모음 (Vowels) - CMUDict ARPAbet 기준
    'AE': np.array([1.0, 0.0, 0.5, 0.0, 0.0]),  # as in 'cat' (æ)
    'AH': np.array([0.9, 0.1, 0.6, 0.0, 0.0]),  # as in 'cut' (ʌ)
    'AY': np.array([0.5, 0.8, 0.1, 0.0, 0.0]), # as in 'mind' (aɪ)
    'EH': np.array([0.4, 0.7, 0.2, 0.0, 0.0]),  # as in 'spend' (ɛ)
    'IY': np.array([0.2, 0.9, 0.1, 0.0, 0.0]),  # as in 'feel' (i)
    'AO': np.array([0.8, 0.0, 0.7, 0.0, 0.0]),  # as in 'talk' (ɔ)
    'R': np.array([0.1, 0.0, 0.8, 0.5, 0.5]),  # 'R'
    'UW': np.array([0.3, 0.6, 0.4, 0.0, 0.0]),  # as in 'food' (u)
    
    # 자음 (Consonants) - CMUDict ARPAbet 기준
    'T': np.array([0.0, 1.0, 0.0, 1.0, 0.0]),  # Consonant 'T'
    'D': np.array([0.0, 0.9, 0.0, 1.0, 0.1]),  # Consonant 'D' (T와 유사)
    'N': np.array([0.0, 0.8, 0.1, 1.0, 0.0]),  # Consonant 'N'
    'K': np.array([0.0, 0.5, 0.0, 0.5, 0.0]),  # Consonant 'K'
    'V': np.array([0.0, 0.8, 0.0, 1.0, 0.2]),  # Consonant 'V'
    'L': np.array([0.0, 0.0, 0.7, 0.4, 0.0]),  # Consonant 'L'
    # 복합 음소도 CMUDict에서 직접 처리될 수 있도록 매핑 (예시: 'nd' 대신 'N' 'D' 조합)
}

# =========================================================
# 2. 핵심 계산 함수 (ARPAbet 기반으로 직접 벡터화하도록 수정)
# =========================================================

def get_phonetic_tail(word):
    """
    CMUDict에서 단어의 끝 음소(clean ARPAbet)를 추출합니다.
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
# 🚨 핵심 수정 부분 1: IPA 변환 함수를 ARPAbet-to-IPA 매핑 로직으로 유지
# (디버깅 정보 출력 및 최종 JSON 출력을 위해 IPA 변환 로직은 유지)
# ---------------------------------------------------------
ARPABET_TO_IPA_MAP = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ', 
    'B': 'b', 'CH': 'ʧ', 'D': 'd', 'DH': 'ð', 'EH': 'ɛ', 'ER': 'əɹ', 
    'EY': 'eɪ', 'F': 'f', 'G': 'g', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 
    'JH': 'ʤ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 
    'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'ʃ', 
    'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v', 'W': 'w', 
    'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
}

def arpabet_to_ipa(arpabet_phons):
    """ARPAbet 음소열을 직접 매핑하여 IPA 문자열로 변환합니다."""
    if not arpabet_phons:
        return None
    
    ipa_phons = [ARPABET_TO_IPA_MAP.get(phon.upper(), '') for phon in arpabet_phons]
    ipa_str = "".join([p for p in ipa_phons if p])
    
    return ipa_str if ipa_str else None
# ---------------------------------------------------------


def calculate_rhyme_score(phon_list1, phon_list2): # 🚨 인자명을 phon_list로 변경
    """두 ARPAbet 음소열의 벡터 유사도 점수 (코사인 유사도)를 계산합니다."""
    
    # 🚨 IPA 대신 ARPAbet 기호를 사용하여 임베딩 벡터를 찾습니다.
    # 길이 불일치 검증은 get_rhyme_candidates_with_score에서 했으므로 생략
    
    vec1_list = [PHONEME_EMBEDDINGS.get(p.upper(), np.zeros(5)) for p in phon_list1]
    vec2_list = [PHONEME_EMBEDDINGS.get(p.upper(), np.zeros(5)) for p in phon_list2]
    
    vec1 = np.concatenate(vec1_list)
    vec2 = np.concatenate(vec2_list)
    
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0

    similarity = 1 - cosine(vec1, vec2)
    return max(0, similarity)


@st.cache_data(show_spinner=False)
def get_rhyme_candidates_with_score(target_word: str, top_n=100):
    """CMUDict 전체를 검색하여 끝 음소가 일치하는 후보를 찾고 점수를 매깁니다."""
    
    target_pron_raw, target_rhyme_unit = get_phonetic_tail(target_word)
    
    if not target_rhyme_unit:
        return {"target_word": target_word, "target_ipa": "N/A", "raw_arpabet": target_pron_raw, "candidates": []}

    # 라임 유닛 IPA 변환 (디버깅 및 JSON 출력용)
    target_ipa = arpabet_to_ipa(target_rhyme_unit)
    
    if not target_ipa:
        return {"target_word": target_word, "target_ipa": "N/A", "raw_arpabet": target_pron_raw, "candidates": []}

    candidates_list = []
    
    for word, pron_list in p_dict.items():
        pron_raw = pron_list[0]
        
        candidate_pron_clean_full = [p.rstrip('0123') for p in pron_raw]
        candidate_rhyme_unit = candidate_pron_clean_full[-len(target_rhyme_unit):]
        
        if word == target_word.lower() or len(word) <= 2 or len(candidate_rhyme_unit) != len(target_rhyme_unit):
            continue
            
        # 2. 끝 음소 일치 확인 (가장 단순한 라임 조건)
        if candidate_rhyme_unit == target_rhyme_unit:
            
            # IPA 변환 (디버깅 및 JSON 출력용)
            candidate_ipa = arpabet_to_ipa(candidate_rhyme_unit) 
            
            if candidate_ipa:
                # 🚨 ARPAbet 기호를 직접 calculate_rhyme_score에 전달하여 점수 계산
                score = calculate_rhyme_score(target_rhyme_unit, candidate_rhyme_unit) 
                
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
st.caption("✅ 강세(Stresses)를 무시하고 단어 끝 3개 음소 기준으로 검색합니다. (기말 프로젝트 최종 개선)")

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
            # 스코어가 0.0이면 제외 (너무 먼 단어)
            if c['score'] == 0.0:
                continue

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
