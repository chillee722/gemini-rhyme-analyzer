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
#    - 이 벡터들을 조정하여 음소 간의 유사도(Score)를 조절할 수 있습니다.
# =========================================================

# PHONEME_EMBEDDINGS의 키는 CMUDict의 ARPAbet 기호이며, 5차원 벡터로 정의됩니다.
PHONEME_EMBEDDINGS = {
    # Vowels (모음): 모음끼리는 유사도를 높이고, 자음과는 낮게 설정됩니다.
    'AE': np.array([1.0, 0.0, 0.5, 0.0, 0.0]),  # 'cat' (æ)
    'AH': np.array([0.9, 0.1, 0.6, 0.0, 0.0]),  # 'cut' (ʌ) - AE와 유사하도록 설정
    'AY': np.array([0.5, 0.8, 0.1, 0.0, 0.0]), # 'mind' (aɪ)
    'EH': np.array([0.4, 0.7, 0.2, 0.0, 0.0]),  # 'spend' (ɛ) - AY와 근사 라임이 되도록 설정
    'IY': np.array([0.2, 0.9, 0.1, 0.0, 0.0]),  # 'feel' (i)
    'AO': np.array([0.8, 0.0, 0.7, 0.0, 0.0]),  # 'talk' (ɔ)
    'R': np.array([0.1, 0.0, 0.8, 0.5, 0.5]),  # 'R'
    'UW': np.array([0.3, 0.6, 0.4, 0.0, 0.0]),  # 'food' (u)
    
    # Consonants (자음): 조음 위치가 같은 자음끼리 유사하도록 설정됩니다.
    'T': np.array([0.0, 1.0, 0.0, 1.0, 0.0]),  # 'T'
    'D': np.array([0.0, 1.0, 0.0, 1.0, 0.1]),  # 'D' (T와 매우 유사함)
    'N': np.array([0.0, 0.9, 0.1, 1.0, 0.0]),  # 'N' (D와 유사함)
    'K': np.array([0.0, 0.5, 0.0, 0.5, 0.0]),  # 'K'
    'V': np.array([0.0, 0.8, 0.0, 1.0, 0.2]),  # 'V'
    'L': np.array([0.0, 0.0, 0.7, 0.4, 0.0]),  # 'L'
}

# =========================================================
# 2. 핵심 계산 함수 (ARPAbet 기반으로 직접 벡터화하도록 수정)
# =========================================================

def get_phonetic_tail(word, rhyme_length):
    """
    CMUDict에서 단어의 끝 음소(clean ARPAbet)를 추출합니다.
    (rhyme_length에 따라 끝 음소의 길이를 결정합니다.)
    """
    word = word.lower()
    if word not in p_dict:
        return None, None
        
    pron_raw = p_dict[word][0] 
    
    # ARPAbet에서 스트레스 마크 제거 (0, 1, 2)
    pron_clean_full = [phon.rstrip('0123') for phon in pron_raw]
    
    # 사용자가 지정한 rhyme_length에 따라 끝 음소를 추출합니다.
    if len(pron_clean_full) >= rhyme_length:
        rhyme_unit_clean = pron_clean_full[-rhyme_length:]
    else:
        # 단어가 짧으면 전체 음소를 사용합니다.
        rhyme_unit_clean = pron_clean_full
    
    # 원본 pron, 클린 라임 유닛을 반환
    return pron_raw, rhyme_unit_clean

# ---------------------------------------------------------
# IPA 변환 함수 (디버깅용)
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


def calculate_rhyme_score(phon_list1, phon_list2):
    """두 ARPAbet 음소열의 벡터 유사도 점수 (코사인 유사도)를 계산합니다."""
    
    # PHONEME_EMBEDDINGS의 키는 ARPAbet 기호입니다.
    
    vec1_list = [PHONEME_EMBEDDINGS.get(p.upper(), np.zeros(5)) for p in phon_list1]
    vec2_list = [PHONEME_EMBEDDINGS.get(p.upper(), np.zeros(5)) for p in phon_list2]
    
    # 길이가 다른 경우를 방지하기 위해 가장 짧은 길이로 잘라줍니다.
    min_len = min(len(vec1_list), len(vec2_list))
    vec1_list = vec1_list[:min_len]
    vec2_list = vec2_list[:min_len]

    vec1 = np.concatenate(vec1_list)
    vec2 = np.concatenate(vec2_list)
    
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0

    similarity = 1 - cosine(vec1, vec2)
    return max(0, similarity)


@st.cache_data(show_spinner=False)
def get_rhyme_candidates_with_score(target_word: str, rhyme_length: int, top_n=100):
    """CMUDict 전체를 검색하여 끝 음소가 일치하는 후보를 찾고 점수를 매깁니다."""
    
    # 1. 대상 단어의 라임 유닛 추출
    target_pron_raw, target_rhyme_unit = get_phonetic_tail(target_word, rhyme_length)
    
    if not target_rhyme_unit:
        return {"target_word": target_word, "target_ipa": "N/A", "raw_arpabet": target_pron_raw, "candidates": []}

    target_ipa = arpabet_to_ipa(target_rhyme_unit)
    
    if not target_ipa:
        return {"target_word": target_word, "target_ipa": "N/A", "raw_arpabet": target_pron_raw, "candidates": []}

    candidates_list = []
    
    # ----------------------------------------------------------------
    # CMUDict 전체 스캔 로직
    # ----------------------------------------------------------------
    for word, pron_list in p_dict.items():
        pron_raw = pron_list[0]
        
        candidate_pron_clean_full = [p.rstrip('0123') for p in pron_raw]
        
        # 1. 단어 필터링 (자기 자신 제외, 너무 짧은 단어 제외)
        if word == target_word.lower() or len(word) <= 2:
            continue
        
        # 2. 라임 유닛 추출 및 길이 확인
        candidate_rhyme_unit = candidate_pron_clean_full[-len(target_rhyme_unit):]
        
        if len(candidate_rhyme_unit) != len(target_rhyme_unit):
            continue
            
        # 3. 끝 음소 일치 확인 (가장 단순한 라임 조건)
        if candidate_rhyme_unit == target_rhyme_unit:
            
            candidate_ipa = arpabet_to_ipa(candidate_rhyme_unit) 
            
            if candidate_ipa:
                # ARPAbet 기호를 직접 calculate_rhyme_score에 전달하여 점수 계산
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
# 3. Streamlit UI (정확도 조정 기능 추가)
# =========================================================

st.set_page_config(page_title="Phonetics Analyzer (Simplified Rhyme)", layout="centered")

st.title("🎤 CMUDict 통합: 음소 임베딩 단순 라임 분석")
st.caption("✅ 기말 프로젝트 최종 개선: 발음 유사도 기준(임베딩 벡터) 및 검색 길이 조정 가능")

st.sidebar.header("🎯 정확도 조정 파라미터")
rhyme_length = st.sidebar.slider(
    "라임 유닛 길이 (음소 개수)", 
    min_value=1, 
    max_value=5, 
    value=3,
    help="검색할 단어의 끝 음소 몇 개를 비교할지 결정합니다. 길수록 라임이 엄격해집니다."
)

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
        # 사용자가 선택한 rhyme_length를 함수에 전달
        analysis_result = get_rhyme_candidates_with_score(input_word, rhyme_length)
    
    # --- 디버깅 정보 출력 ---
    st.markdown("#### 🚨 디버깅 정보 (발표 시 숨김 권장)")
    st.markdown(f"**CMUDict 원본 발음 (ARPAbet):** `{analysis_result.get('raw_arpabet')}`")
    st.markdown(f"**대상 단어 IPA (라임 유닛):** `{analysis_result['target_ipa']}` (비교 길이: {rhyme_length}개 음소)")
    
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
        st.warning(f"CMUDict에서 '{input_word}'에 대한 라임 유닛을 찾지 못했거나, 일치하는 끝 {rhyme_length}개 음소 단어를 찾지 못했습니다. (비교 길이를 줄여보세요.)")

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
