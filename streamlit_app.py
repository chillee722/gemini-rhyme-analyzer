import streamlit as st
import numpy as np
from scipy.spatial.distance import cosine
import json
import nltk
from nltk.corpus import cmudict
import re
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
# 1. 음소 임베딩 (Slant Rhyme 계산용 - 필수 아님, 보조 수단)
# =========================================================

# 실제 라임은 강세 모음 일치로 판단되므로, 임베딩은 매우 단순화됩니다.
PHONEME_EMBEDDINGS: Dict[str, np.ndarray] = {
    # Vowels (모음):
    'AE': np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    'IY': np.array([0.9, 0.1, 0.0, 0.0, 0.0]),
    'R': np.array([0.1, 0.0, 0.8, 0.5, 0.5]),
    'L': np.array([0.0, 0.0, 0.7, 0.4, 0.0]),
    'T': np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
    'D': np.array([0.0, 1.0, 0.0, 1.0, 0.1]),
    'K': np.array([0.0, 0.5, 0.0, 0.5, 0.0]),
    'AH': np.array([0.9, 0.1, 0.6, 0.0, 0.0]),
    'ER': np.array([0.5, 0.0, 0.8, 0.5, 0.5]), # 'R-colored' vowel
}
# 모든 ARPAbet 기호가 정의되지 않았을 경우를 대비한 맵 (실제 CMUDict는 40개 이상임)
def get_embedding(phon: str) -> np.ndarray:
    return PHONEME_EMBEDDINGS.get(phon.upper(), np.zeros(5))

# =========================================================
# 2. 핵심 수정 함수: 라임 유닛 추출 (강세 모음 기준)
# =========================================================

def get_rhyme_unit(word: str) -> Optional[Tuple[List[str], List[str], List[str]]]:
    """
    단어의 발음에서 마지막 강세 모음을 기준으로 라임 유닛을 추출합니다.
    """
    word = word.lower()
    if word not in p_dict:
        return None
        
    pron_raw = p_dict[word][0]
    
    # 마지막 강세 모음 (1차 '1' 또는 2차 '2')의 인덱스를 찾습니다.
    # CMUDict 발음에서 모음 끝에 1 또는 2가 붙은 것을 찾습니다.
    stress_indices = [i for i, phon in enumerate(pron_raw) if phon[-1] in ('1', '2')]
    
    # 1. 라임 유닛 시작점 결정 (강세 모음)
    if not stress_indices:
        # 강세가 없는 단어(예: 'a', 'the')는 단어 전체를 라임 유닛으로 간주 (첫 음소부터)
        start_index = 0
    else:
        # 가장 마지막 강세 위치를 라임 유닛의 시작점으로 삼습니다.
        start_index = stress_indices[-1]
        
    # 2. 라임 유닛 분리
    # Onset (강세 모음 앞) - 라임이 될 수 없는 부분
    onset_raw = pron_raw[:start_index] 
    # Rhyme Unit (강세 모음부터 끝까지) - 라임이 되어야 하는 부분
    rhyme_unit_raw = pron_raw[start_index:]
    
    # 3. 스트레스 마크 제거 (순수 음소만 남깁니다.)
    onset_clean = [phon.rstrip('0123') for phon in onset_raw]
    rhyme_unit_clean = [phon.rstrip('0123') for phon in rhyme_unit_raw]
    
    # 원본 pron, Onset(앞부분), Rhyme Unit(라임 부분) 반환
    return pron_raw, onset_clean, rhyme_unit_clean

# ---------------------------------------------------------
# IPA 변환 함수 (디버깅 및 표시용)
# ---------------------------------------------------------
ARPABET_TO_IPA_MAP = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ', 'B': 'b', 'CH': 'ʧ', 'D': 'd', 'DH': 'ð', 'EH': 'ɛ', 'ER': 'əɹ', 'EY': 'eɪ', 'F': 'f', 'G': 'g', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'ʤ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
}

def arpabet_to_ipa(arpabet_phons: List[str]) -> Optional[str]:
    """ARPAbet 음소열을 직접 매핑하여 IPA 문자열로 변환합니다."""
    if not arpabet_phons:
        return None
    
    ipa_phons = [ARPABET_TO_IPA_MAP.get(phon.upper(), '') for phon in arpabet_phons]
    ipa_str = "".join([p for p in ipa_phons if p])
    
    return ipa_str if ipa_str else None


def calculate_rhyme_score_slant(phon_list1: List[str], phon_list2: List[str]) -> float:
    """
    두 ARPAbet 음소열의 벡터 유사도 점수 (코사인 유사도)를 계산합니다. 
    (불완전한 라임(Slant Rhyme)을 찾기 위한 보조 수단으로 사용)
    """
    
    # 길이가 다른 경우를 대비하여 가장 짧은 길이로 잘라줍니다.
    # (주의: 완벽한 라임 검색에서는 길이가 같아야 함)
    min_len = min(len(phon_list1), len(phon_list2))
    
    vec1_list = [get_embedding(p) for p in phon_list1[:min_len]]
    vec2_list = [get_embedding(p) for p in phon_list2[:min_len]]
    
    if not vec1_list or not vec2_list:
        return 0.0

    vec1 = np.concatenate(vec1_list)
    vec2 = np.concatenate(vec2_list)
    
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0

    # 코사인 유사도 계산: 1에 가까울수록 유사함
    similarity = 1 - cosine(vec1, vec2)
    return max(0, similarity)


@st.cache_data(show_spinner=False)
def get_rhyme_candidates_with_score(target_word: str, top_n=100) -> Dict:
    """CMUDict 전체를 검색하여 라임 유닛이 일치하는 후보를 찾고 점수를 매깁니다."""
    
    # 1. 대상 단어의 라임 유닛 추출 (강세 모음 기준)
    target_info = get_rhyme_unit(target_word)
    
    if not target_info:
        return {"target_word": target_word, "target_ipa": "N/A", "raw_arpabet": "N/A", "candidates": []}

    target_pron_raw, target_onset, target_rhyme_unit = target_info
    target_ipa = arpabet_to_ipa(target_rhyme_unit)
    target_rhyme_len = len(target_rhyme_unit)
    
    if not target_ipa or target_rhyme_len == 0:
        return {"target_word": target_word, "target_ipa": "N/A", "raw_arpabet": target_pron_raw, "candidates": []}

    candidates_list = []
    
    # ----------------------------------------------------------------
    # CMUDict 전체 스캔 로직
    # ----------------------------------------------------------------
    for word, _ in p_dict.items():
        
        candidate_info = get_rhyme_unit(word)
        
        if not candidate_info:
            continue
            
        candidate_pron_raw, candidate_onset, candidate_rhyme_unit = candidate_info
        
        # 1. 단어 필터링 (자기 자신 제외, 너무 짧은 단어 제외)
        if word == target_word.lower() or len(word) <= 2:
            continue
            
        # 2. Onset (강세 모음 앞)이 같으면 안 됨 (동일한 단어/파생어 제외)
        # 예: 'cat' ('K AE1 T')과 'un-cat' ('AH0 N K AE1 T')이 라임이 될 수는 있으나,
        # 일반적으로 첫 자음 소리가 달라야 합니다. (완벽한 라임 조건)
        if len(target_onset) > 0 and len(candidate_onset) > 0 and target_onset[-1] == candidate_onset[-1]:
             # 마지막 자음이 같으면 보통 라임에서 제외 (예: 'can'과 'scan' 같은 경우)
             pass 
            
        # 3. 라임 유닛 길이 확인 (핵심)
        if len(candidate_rhyme_unit) != target_rhyme_len:
            continue
        
        # 4. 점수 계산
        candidate_ipa = arpabet_to_ipa(candidate_rhyme_unit)
        score = 0.0
        rhyme_type = "Poor Match/Error"
        
        # A. Perfect Rhyme (완벽한 라임): 음소열이 완전히 일치
        if candidate_rhyme_unit == target_rhyme_unit:
            score = 1.0 
            rhyme_type = "Perfect Rhyme (True Rhyme)"
            
            # 마지막 Onset 자음이 일치하면 (예: cat/sat) 완벽한 라임
            # Onset의 마지막 자음이 서로 다른지 확인 (c/s)
            if len(target_onset) > 0 and len(candidate_onset) > 0 and target_onset[-1] == candidate_onset[-1]:
                 # 같은 자음이면 매우 유사한 소리, 하지만 라임은 아님 (예: 'pat'/'pad'의 모음 앞 자음 P/P)
                 pass
            
        # B. Slant Rhyme (불완전 라임): 벡터 유사도를 통한 근접 라임 찾기 (보조)
        elif candidate_ipa:
            score = calculate_rhyme_score_slant(target_rhyme_unit, candidate_rhyme_unit)
            if score >= 0.70:
                 rhyme_type = "Slant Rhyme (Near Match)"
            else:
                 continue # 점수가 너무 낮으면 제외
        
        if score > 0.0:
            candidates_list.append({
                "word": word,
                "score": round(score, 4), # 소수점 4자리까지 표시
                "ipa": candidate_ipa,
                "rhyme_unit": " ".join(candidate_rhyme_unit),
                "rhyme_type": rhyme_type
            })

    # 정렬 및 결과 반환
    candidates_list.sort(key=lambda x: x['score'], reverse=True)

    return {
        "target_word": target_word,
        "target_ipa": target_ipa,
        "target_rhyme_unit": " ".join(target_rhyme_unit),
        "raw_arpabet": target_pron_raw,
        "candidates": candidates_list[:top_n]
    }


# =========================================================
# 3. Streamlit UI 
# =========================================================

st.set_page_config(page_title="Phonetics Analyzer (Rhyme Analysis)", layout="centered")

st.title("🎤 CMUDict 기반 라임 분석기 (강세 모음 기준)")
st.caption("✅ **강세 모음**을 기준으로 라임 유닛을 추출하도록 수정되었습니다. (완벽 라임 우선 검색)")

# 사용자 입력
input_word = st.text_input("분석할 단어를 입력하세요 (예: cat, together, compute)", "together")

if input_word:
    st.subheader(f"🔍 '{input_word}'에 대한 분석 결과")
    
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
        st.warning(f"CMUDict에서 '{input_word}'에 대한 적절한 라임 후보를 찾지 못했습니다.")

    # 3. 최종 API 응답 (JSON)
    st.markdown("---")
    st.markdown("#### 🤖 Gemini에게 제공할 최종 API 응답 (JSON)")
    final_json_output = {
        "target_word": analysis_result["target_word"],
        "target_ipa": analysis_result["target_ipa"],
        "candidates": [{k: v for k, v in c.items() if k not in ['rhyme_unit']} for c in analysis_result["candidates"]]
    }
    st.code(json.dumps(final_json_output, indent=2), language='json')
