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
    
    # 스트레스 마크 제거
    onset_clean = [phon.rstrip('0123') for phon in onset_raw]
    rhyme_unit_clean = [phon.rstrip('0123') for phon in rhyme_unit_raw]

    # 반환: (원본 발음, Onset, Rhyme Unit)
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

    # 유사도가 높을수록 1에 가깝습니다.
    return max(0, 1 - cosine(vec1, vec2))


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
    """Front Rhyme 점수를 포함하여 라임 후보를 찾습니다."""
    
    target_info = get_rhyme_unit(target_word)
    
    if not target_info:
        return {"target_word": target_word, "target_ipa": "N/A", "target_rhyme_unit": "N/A", "raw_arpabet": "N/A", "candidates": []}

    target_pron_raw, target_onset, target_rhyme_unit = target_info
    target_ipa = arpabet_to_ipa(target_rhyme_unit)
    target_vowel = target_rhyme_unit[0] if target_rhyme_unit else ""
    target_rhyme_len = len(target_rhyme_unit)
    
    if not target_ipa or not target_rhyme_unit:
        return {"target_word": target_word, "target_ipa": "N/A", "target_rhyme_unit": "N/A", "raw_arpabet": target_pron_raw, "candidates": []}

    candidates_list = []
    
    # --- Front Rhyme 구현을 위한 파라미터 ---
    # Rhyme Unit 유사도(Slant Score)에 Front Rhyme 유사도를 합산할 비율 (가중치)
    FRONT_RHYME_WEIGHT = 0.1 
    # ------------------------------------------

    for word, _ in p_dict.items():
        
        candidate_info = get_rhyme_unit(word)
        if not candidate_info:
            continue
            
        candidate_pron_raw, candidate_onset, candidate_rhyme_unit = candidate_info
        
        if word == target_word.lower():
            continue
            
        score = 0.0
        rhyme_type = "Slant/Poor Match"
        candidate_vowel = candidate_rhyme_unit[0] if candidate_rhyme_unit else ""
        
        # A. Perfect Rhyme 검사 (완벽 라임은 제외)
        is_perfect = False
        if len(candidate_rhyme_unit) == target_rhyme_len and candidate_rhyme_unit == target_rhyme_unit:
            is_onset_different = (not target_onset or not candidate_onset or target_onset[-1] != candidate_onset[-1])
            if is_onset_different:
                is_perfect = True

        if is_perfect:
            continue # 완벽 라임 제외

        # B. Slant Rhyme 및 Front Rhyme 구현
        len_diff = abs(len(candidate_rhyme_unit) - target_rhyme_len)
        
        if len_diff <= 2: 
            
            # 1. Rhyme Unit 유사도 계산 (메인)
            slant_score_base = calculate_slant_score(target_rhyme_unit, candidate_rhyme_unit, target_vowel, candidate_vowel)
            
            # 2. Front Rhyme 유사도 계산 (보너스)
            front_rhyme_similarity = 0.0
            if target_onset and candidate_onset:
                front_rhyme_similarity = calculate_front_rhyme_score(target_onset, candidate_onset)
                
            # 3. 최종 점수: Rhyme Unit 유사도 + Front Rhyme 보너스
            # (총점은 1.0으로 다시 캡을 씌워 오버플로우 방지)
            final_weighted_score = slant_score_base + (front_rhyme_similarity * FRONT_RHYME_WEIGHT)
            score = min(1.0, final_weighted_score)
            
            # 등급 분류
            if score >= 0.95: 
                rhyme_type = "Multi-Syllable Slant/Front Rhyme (Near Perfect)"
            elif score >= 0.85:
                rhyme_type = "Slant/Front Rhyme (Good Match)"
            else:
                continue 
        else:
            continue 

        if score > 0.0:
            candidates_list.append({
                "word": word,
                "score": round(score, 4), 
                "ipa": arpabet_to_ipa(candidate_rhyme_unit),
                "rhyme_unit": " ".join(candidate_rhyme_unit),
                "rhyme_type": rhyme_type
            })

    candidates_list.sort(key=lambda x: x['score'], reverse=True)

    return {
        "target_word": target_word,
        "target_ipa": target_ipa,
        "target_rhyme_unit": " ".join(target_rhyme_unit),
        "raw_arpabet": target_pron_raw,
        "candidates": candidates_list[:top_n]
    }


# =========================================================
# 3. Streamlit UI (최종)
# =========================================================

st.set_page_config(page_title="Phonetics Analyzer (Front Rhyme Final)", layout="centered")

st.title("🎤 CMUDict 통합: 에미넴 스타일 고급 라임 분석기 (최종)")
st.caption("✅ **Front Rhyme(두음 유사도)을 명시적으로 계산**하고 점수에 반영하여 복합 라임을 구현합니다. Score 버그 수정 완료.")

# 사용자 입력
input_word = st.text_input("분석할 단어를 입력하세요 (예: together, recently, lawyer)", "recently")

if input_word:
    st.subheader(f"🔍 '{input_word}'에 대한 CMUDict 기반 분석 결과")
    
    # 계산 로직 실행
    with st.spinner('CMUDict를 스캔하고 라임 유닛을 계산 중...'):
        analysis_result = get_rhyme_candidates_with_score(input_word)
    
    # --- 디버깅 정보 출력 ---
    st.markdown("#### 🚨 디버깅 및 분석 정보")
    st.markdown(f"**CMUDict 원본 발음 (ARPAbet):** `{analysis_result.get('raw_arpabet')}`")
    st.markdown(f"**대상 라임 유닛 (ARPAbet):** `{analysis_result['target_rhyme_unit']}`")
    st.markdown(f"**대상 라임 유닛 (IPA):** `{analysis_result['target_ipa']}`")
    
    # 2. 유사도 테이블 표시
    st.markdown("---")
    st.markdown("#### CMUDict 기반 검색 결과 (점수 순 정렬)")
    
    if analysis_result['candidates']:
        # 테이블 데이터 준비
        data = []
        for c in analysis_result['candidates']:
            data.append({
                "Word": c['word'],
                "Rhyme Unit (ARPAbet)": c['rhyme_unit'],
                "IPA": c['ipa'],
                "Score": f"{c['score']:.4f}",
                "Rhyme Type": c['rhyme_type']
            })
        
        st.dataframe(data, use_container_width=True, hide_index=True)
    else:
        st.warning(f"CMUDict에서 '{input_word}'에 대한 적절한 슬랜트 라임 후보를 찾지 못했습니다. (완벽 라임 제외)")

    # 3. Gemini가 받을 API 응답 (JSON)
    st.markdown("---")
    st.markdown("#### 🤖 Gemini에게 제공할 최종 API 응답 (JSON)")
    final_json_output = {
        "target_word": analysis_result["target_word"],
        "target_ipa": analysis_result["target_ipa"],
        "candidates": [{k: v for k, v in c.items() if k not in ['rhyme_unit']} for c in analysis_result["candidates"]]
    }
    st.code(json.dumps(final_json_output, indent=2), language='json')
