import streamlit as st
import numpy as np
from scipy.spatial.distance import cosine
import json
import nltk
from nltk.corpus import cmudict
from typing import List, Tuple, Optional, Dict

# --- CMUDict 다운로드 및 로드 ---
@st.cache_resource(show_spinner="CMUDict 사전을 로드 중...")
def load_cmudict():
    """CMUDict를 로드하고 필요한 경우 다운로드합니다."""
    try:
        nltk.data.find('corpora/cmudict')
    except LookupError:
        nltk.download('cmudict')
        
    return cmudict.dict()

p_dict = load_cmudict()

# =========================================================
# 1. 음소 임베딩 (최종 버전 유지)
# =========================================================

# ARPAbet 기호에 언어학적 특징 반영: [모음, 비음, 발성, 조음위치, 조음방법]
PHONEME_EMBEDDINGS: Dict[str, np.ndarray] = {
    # Vowels
    'AE': np.array([1.0, 0.1, 1.0, 0.2, 0.0]), 'IY': np.array([0.95, 0.1, 1.0, 0.0, 0.0]), 
    'IH': np.array([0.9, 0.1, 1.0, 0.1, 0.0]), 'AH': np.array([0.7, 0.2, 1.0, 0.5, 0.0]),
    'AY': np.array([0.9, 0.1, 1.0, 0.3, 0.1]), 'AO': np.array([0.9, 0.1, 1.0, 0.7, 0.1]),
    'OW': np.array([0.85, 0.1, 1.0, 0.8, 0.1]), 'AW': np.array([0.85, 0.1, 1.0, 0.6, 0.1]),
    'ER': np.array([0.6, 0.0, 1.0, 0.5, 0.5]),
    
    # Consonants (Flow, Consonance 강화)
    'T': np.array([0.0, 0.75, 0.0, 0.4, 0.9]), 'D': np.array([0.0, 0.75, 1.0, 0.4, 0.9]), 
    'S': np.array([0.0, 0.0, 0.0, 0.35, 0.7]), 'Z': np.array([0.0, 0.0, 1.0, 0.35, 0.7]),
    'N': np.array([0.0, 0.9, 1.0, 0.4, 0.8]), 'M': np.array([0.0, 0.9, 1.0, 0.2, 0.8]), 
    'K': np.array([0.0, 0.5, 0.0, 0.8, 0.9]), 'G': np.array([0.0, 0.5, 1.0, 0.8, 0.9]),
    'F': np.array([0.0, 0.5, 0.0, 0.1, 0.7]), 'V': np.array([0.0, 0.5, 1.0, 0.1, 0.7]),

    # L/R/Y/W 계열 (활음/유음 유사성 극대화)
    'L': np.array([0.6, 0.0, 1.0, 0.7, 0.3]), 'R': np.array([0.6, 0.0, 1.0, 0.5, 0.4]),
    'Y': np.array([0.7, 0.0, 1.0, 0.1, 0.1]), 'W': np.array([0.7, 0.0, 1.0, 0.9, 0.1]), 
}

def get_embedding(phon: str) -> np.ndarray:
    """정의된 음소 임베딩을 가져오고, 없으면 0 벡터를 반환합니다."""
    return PHONEME_EMBEDDINGS.get(phon.upper(), np.zeros(5))

# =========================================================
# 2. 핵심 함수: 라임 유닛 추출 및 점수 계산
# =========================================================

def get_rhyme_unit(word: str) -> Optional[Tuple[List[str], List[str], List[str]]]:
    """단어의 마지막 강세 모음(1 또는 2)을 기준으로 Rhyme Unit과 Onset을 추출합니다."""
    word = word.lower()
    if word not in p_dict:
        return None
        
    pron_raw = p_dict[word][0]
    stress_markers = ['1', '2']
    stress_indices = [i for i, phon in enumerate(pron_raw) if phon[-1] in stress_markers]
    
    if not stress_indices:
        start_index = 0
    else:
        start_index = stress_indices[-1]
        
    onset_raw = pron_raw[:start_index] 
    rhyme_unit_raw = pron_raw[start_index:]
    
    onset_clean = [phon.rstrip('0123') for phon in onset_raw]
    rhyme_unit_clean = [phon.rstrip('0123') for phon in rhyme_unit_raw]

    return pron_raw, onset_clean, rhyme_unit_clean


def calculate_front_rhyme_score(onset_list1: List[str], onset_list2: List[str]) -> float:
    """Onset(두음) 간의 코사인 유사도를 계산하여 Front Rhyme 점수를 산출합니다."""
    
    if not onset_list1 or not onset_list2:
        return 0.0
        
    # 비교를 위해 가장 짧은 Onset 길이를 기준으로 맞춥니다.
    min_len = min(len(onset_list1), len(onset_list2))
    
    vec1_list = [get_embedding(p) for p in onset_list1[-min_len:]]
    vec2_list = [get_embedding(p) for p in onset_list2[-min_len:]]
    
    vec1 = np.concatenate(vec1_list)
    vec2 = np.concatenate(vec2_list)
    
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0

    return max(0, 1 - cosine(vec1, vec2))


def arpabet_to_ipa(arpabet_phons: List[str]) -> Optional[str]:
    """ARPAbet 음소열을 IPA 문자열로 변환합니다."""
    ARPABET_TO_IPA_MAP = {
        'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ', 'B': 'b', 'CH': 'ʧ', 'D': 'd', 'DH': 'ð', 'EH': 'ɛ', 'ER': 'əɹ', 'EY': 'eɪ', 'F': 'f', 'G': 'g', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'ʤ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
    }
    ipa_phons = [ARPABET_TO_IPA_MAP.get(phon.upper(), '') for phon in arpabet_phons]
    return "".join([p for p in ipa_phons if p])


def calculate_slant_score(phon_list1: List[str], phon_list2: List[str], target_vowel: str, candidate_vowel: str) -> float:
    """Rhyme Unit 간의 유사도를 계산합니다. (Score > 1.0 버그 수정됨)"""
    len1, len2 = len(phon_list1), len(phon_list2)
    max_len = max(len1, len2)
    
    vec1_list = [get_embedding(p) for p in phon_list1]
    vec2_list = [get_embedding(p) for p in phon_list2]
    
    vec1 = np.concatenate(vec1_list + [np.zeros(5)] * (max_len - len1))
    vec2 = np.concatenate(vec2_list + [np.zeros(5)] * (max_len - len2))
    
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0

    similarity = 1 - cosine(vec1, vec2)
    
    vowel_bonus = 0.0
    if target_vowel and target_vowel == candidate_vowel:
        vowel_bonus = 0.05 

    final_score = similarity + vowel_bonus
    # Score > 1.0 버그 수정: 최대값을 1.0으로 설정
    return min(1.0, max(0, final_score))


@st.cache_data(show_spinner=False)
def get_rhyme_candidates_with_score(target_word: str, top_n=100) -> Dict:
    """라임 유형별로 분류하여 후보 단어를 출력합니다."""
    
    target_info = get_rhyme_unit(target_word)
    
    if not target_info:
        return {"target_word": target_word, "target_ipa": "N/A", "target_rhyme_unit": "N/A", "raw_arpabet": "N/A", "candidates": []}

    target_pron_raw, target_onset, target_rhyme_unit = target_info
    target_ipa = arpabet_to_ipa(target_rhyme_unit)
    target_vowel = target_rhyme_unit[0] if target_rhyme_unit else ""
    target_rhyme_len = len(target_rhyme_unit)
    
    if not target_ipa or not target_rhyme_unit:
        return {"target_word": target_word, "target_ipa": "N/A", "target_rhyme_unit": "N/A", "raw_arpabet": target_pron_raw, "candidates": []}

    # 최종 출력을 위한 분류 딕셔너리
    classified_candidates = {
        "holorhymes": [], # Front Rhyme + End Rhyme 모두 높은 점수
        "front_rhymes": [], # Front Rhyme만 높은 점수
        "end_rhymes": [], # End Rhyme만 높은 점수
    }
    
    # 점수 기준 (임계값 설정)
    RHYME_THRESHOLD = 0.85
    
    for word, _ in p_dict.items():
        
        candidate_info = get_rhyme_unit(word)
        if not candidate_info:
            continue
            
        candidate_pron_raw, candidate_onset, candidate_rhyme_unit = candidate_info
        
        if word == target_word.lower():
            continue
            
        # A. Perfect Rhyme 검사 (완벽 라임은 제외)
        if len(candidate_rhyme_unit) == target_rhyme_len and candidate_rhyme_unit == target_rhyme_unit:
            is_onset_different = (not target_onset or not candidate_onset or target_onset[-1] != candidate_onset[-1])
            if is_onset_different:
                continue # 완벽 라임 제외

        # B. Front/End Rhyme 점수 계산
        
        # 1. End Rhyme Score (Rhyme Unit 유사도)
        end_score = calculate_slant_score(target_rhyme_unit, candidate_rhyme_unit, target_vowel, candidate_vowel)
        
        # 2. Front Rhyme Score (Onset 유사도)
        front_score = 0.0
        if target_onset and candidate_onset:
            front_score = calculate_front_rhyme_score(target_onset, candidate_onset)
        
        # 3. 분류
        is_front_match = front_score >= RHYME_THRESHOLD
        is_end_match = end_score >= RHYME_THRESHOLD
        
        if is_front_match and is_end_match:
            # Holorime / Mosaic Rhyme (양쪽 모두 높은 점수)
            classified_candidates["holorhymes"].append({
                "word": word,
                "end_score": round(end_score, 4),
                "front_score": round(front_score, 4),
                "ipa": arpabet_to_ipa(candidate_rhyme_unit),
                "rhyme_unit": " ".join(candidate_rhyme_unit),
            })
        elif is_front_match:
            # Front Rhyme (앞부분만 높은 점수)
            classified_candidates["front_rhymes"].append({
                "word": word,
                "front_score": round(front_score, 4),
                "end_score": round(end_score, 4),
                "ipa": arpabet_to_ipa(candidate_rhyme_unit),
                "rhyme_unit": " ".join(candidate_rhyme_unit),
            })
        elif is_end_match:
            # End Rhyme (뒷부분만 높은 점수)
            classified_candidates["end_rhymes"].append({
                "word": word,
                "end_score": round(end_score, 4),
                "front_score": round(front_score, 4),
                "ipa": arpabet_to_ipa(candidate_rhyme_unit),
                "rhyme_unit": " ".join(candidate_rhyme_unit),
            })

    # 분류된 목록을 점수(End Score) 기준으로 정렬
    for key in classified_candidates:
        classified_candidates[key].sort(key=lambda x: x['end_score'], reverse=True)


    # 최종 출력을 위해 분류된 결과를 JSON으로 재구성
    final_output = {
        "target_word": target_word,
        "target_ipa": target_ipa,
        "target_rhyme_unit": " ".join(target_rhyme_unit),
        "classified_rhymes": classified_candidates
    }
    
    return final_output


# =========================================================
# 3. Streamlit UI (최종)
# =========================================================

st.set_page_config(page_title="Phonetics Analyzer (Rhyme Classification)", layout="wide")

st.title("🎤 CMUDict 통합: 에미넴 스타일 라임 분류 분석기")
st.caption("✅ Front Rhyme과 End Rhyme을 **독립적인 점수**로 계산하여 복합 라임 구성을 돕습니다.")

# 사용자 입력
input_word = st.text_input("분석할 단어를 입력하세요 (예: lawyer, nervous, controversy)", "nervous")

if input_word:
    st.subheader(f"🔍 '{input_word}'에 대한 CMUDict 기반 분석 결과")
    
    with st.spinner('CMUDict를 스캔하고 라임 유닛을 계산 중...'):
        analysis_result = get_rhyme_candidates_with_score(input_word)
    
    # --- 디버깅 정보 출력 ---
    st.markdown("#### 🚨 디버깅 및 분석 정보")
    st.markdown(f"**대상 라임 유닛 (IPA):** `{analysis_result['target_ipa']}`")
    
    st.markdown("---")
    st.markdown("#### 🏆 라임 분류 결과 (End Score 순 정렬)")
    
    # 1. Holorime / Mosaic Rhyme 출력
    st.markdown("##### 1. Holorime / Mosaic Rhyme (전체 소리 블록 유사성)")
    if analysis_result['classified_rhymes']['holorhymes']:
        st.dataframe(analysis_result['classified_rhymes']['holorhymes'], use_container_width=True, hide_index=True)
    else:
        st.info("해당 기준을 충족하는 복합 라임 후보가 없습니다.")

    # 2. Front Rhyme 출력
    st.markdown("##### 2. Front Rhyme (두음 유사성: Chain Rhyme에 적합)")
    if analysis_result['classified_rhymes']['front_rhymes']:
        st.dataframe(analysis_result['classified_rhymes']['front_rhymes'], use_container_width=True, hide_index=True)
    else:
        st.info("해당 기준을 충족하는 Front Rhyme 후보가 없습니다.")
    
    # 3. End Rhyme 출력
    st.markdown("##### 3. End Rhyme (끝소리 유사성: 전통적인 Slant Rhyme)")
    if analysis_result['classified_rhymes']['end_rhymes']:
        st.dataframe(analysis_result['classified_rhymes']['end_rhymes'], use_container_width=True, hide_index=True)
    else:
        st.info("해당 기준을 충족하는 End Rhyme 후보가 없습니다.")

    # 4. JSON 출력
    st.markdown("---")
    st.markdown("#### 🤖 Gemini에게 제공할 최종 API 응답 (JSON)")
    st.code(json.dumps(analysis_result, indent=2), language='json')
