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
# 1. 음소 임베딩 (Slant Rhyme 계산용: 모음/자음 유사도 강화)
# =========================================================

# ARPAbet 기호에 언어학적 특징을 반영한 벡터 정의 (예시: 5차원)
PHONEME_EMBEDDINGS: Dict[str, np.ndarray] = {
    # Vowels (모음): 
    'AE': np.array([1.0, 0.1, 0.2, 0.0, 1.0]),  # 'cat' (æ)
    'IY': np.array([0.9, 0.1, 0.0, 0.0, 1.0]),  # 'feel' (i)
    'IH': np.array([0.8, 0.1, 0.1, 0.0, 1.0]),  # 'sit' (ɪ) - IY와 유사
    'AH': np.array([0.7, 0.2, 0.5, 0.0, 1.0]),  # 'cut' (ʌ) 
    'ER': np.array([0.6, 0.0, 0.5, 0.5, 1.0]),  # 'R-colored' vowel
    
    # Consonants (자음): 
    'T': np.array([0.0, 0.8, 0.2, 0.9, 0.0]),
    'D': np.array([0.0, 0.8, 0.2, 0.9, 1.0]), 
    'N': np.array([0.0, 0.9, 0.2, 0.8, 1.0]),
    'K': np.array([0.0, 0.5, 0.8, 0.9, 0.0]),
    'G': np.array([0.0, 0.5, 0.8, 0.9, 1.0]),
    'L': np.array([0.0, 0.0, 0.7, 0.4, 1.0]),
    'M': np.array([0.0, 0.9, 0.1, 0.8, 1.0]),
    'W': np.array([0.3, 0.0, 0.9, 0.3, 1.0]),
    'Y': np.array([0.3, 0.0, 0.1, 0.3, 1.0]),
    'R': np.array([0.1, 0.0, 0.5, 0.5, 1.0]),
    'S': np.array([0.0, 0.0, 0.2, 0.7, 0.0]),
}

def get_embedding(phon: str) -> np.ndarray:
    """정의된 음소 임베딩을 가져오고, 없으면 0 벡터를 반환합니다."""
    return PHONEME_EMBEDDINGS.get(phon.upper(), np.zeros(5))

# =========================================================
# 2. 핵심 수정 함수: 라임 유닛 추출 (강세 모음 기준)
# =========================================================

def get_rhyme_unit(word: str) -> Optional[Tuple[List[str], List[str], List[str], str]]:
    """
    단어의 발음에서 마지막 강세 모음을 기준으로 라임 유닛을 추출합니다.
    (반환: 원본 발음, Onset(두음), Rhyme Unit(운), Stressed Vowel)
    """
    word = word.lower()
    if word not in p_dict:
        return None
        
    pron_raw = p_dict[word][0]
    
    # 마지막 강세 모음 (1차 '1' 또는 2차 '2')의 인덱스를 찾습니다.
    stress_markers = ['1', '2']
    stress_indices = [i for i, phon in enumerate(pron_raw) if phon[-1] in stress_markers]
    
    if not stress_indices:
        # 강세가 없는 단어는 전체를 라임 유닛으로 간주 (첫 음소부터)
        start_index = 0
    else:
        # 가장 마지막 강세 위치를 라임 유닛의 시작점으로 삼습니다.
        start_index = stress_indices[-1]
        
    # 라임 유닛 분리
    onset_raw = pron_raw[:start_index] 
    rhyme_unit_raw = pron_raw[start_index:]
    
    stressed_vowel_raw = rhyme_unit_raw[0] if rhyme_unit_raw else ""
    
    # 스트레스 마크 제거 (순수 음소만 남깁니다.)
    onset_clean = [phon.rstrip('0123') for phon in onset_raw]
    rhyme_unit_clean = [phon.rstrip('0123') for phon in rhyme_unit_raw]
    stressed_vowel_clean = stressed_vowel_raw.rstrip('0123')

    return pron_raw, onset_clean, rhyme_unit_clean, stressed_vowel_clean

# ---------------------------------------------------------
# IPA 변환 함수 (디버깅 및 표시용)
# ---------------------------------------------------------
ARPABET_TO_IPA_MAP = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ', 'B': 'b', 'CH': 'ʧ', 'D': 'd', 'DH': 'ð', 'EH': 'ɛ', 'ER': 'əɹ', 'EY': 'eɪ', 'F': 'f', 'G': 'g', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'ʤ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
}

def arpabet_to_ipa(arpabet_phons: List[str]) -> Optional[str]:
    """ARPAbet 음소열을 IPA 문자열로 변환합니다."""
    ipa_phons = [ARPABET_TO_IPA_MAP.get(phon.upper(), '') for phon in arpabet_phons]
    return "".join([p for p in ipa_phons if p])

def calculate_slant_score(phon_list1: List[str], phon_list2: List[str]) -> float:
    """두 음소열의 코사인 유사도를 계산합니다. (길이가 다르면 0 벡터로 채워서 계산)"""
    
    len1, len2 = len(phon_list1), len(phon_list2)
    max_len = max(len1, len2)
    
    vec1_list = [get_embedding(p) for p in phon_list1]
    vec2_list = [get_embedding(p) for p in phon_list2]
    
    # 길이가 다를 경우, 짧은 쪽을 0 벡터로 채워서 길이를 맞춥니다.
    vec1 = np.concatenate(vec1_list + [np.zeros(5)] * (max_len - len1))
    vec2 = np.concatenate(vec2_list + [np.zeros(5)] * (max_len - len2))
    
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0

    similarity = 1 - cosine(vec1, vec2)
    return max(0, similarity)


@st.cache_data(show_spinner=False)
def get_rhyme_candidates_with_score(target_word: str, top_n=100) -> Dict:
    """강세 모음 및 슬랜트 라임 규칙을 적용하여 라임 후보를 찾습니다."""
    
    target_info = get_rhyme_unit(target_word)
    
    if not target_info:
        return {"target_word": target_word, "target_ipa": "N/A", "target_rhyme_unit": "N/A", "raw_arpabet": "N/A", "candidates": []}

    target_pron_raw, target_onset, target_rhyme_unit, target_vowel = target_info
    target_ipa = arpabet_to_ipa(target_rhyme_unit)
    
    if not target_ipa or not target_rhyme_unit:
        return {"target_word": target_word, "target_ipa": "N/A", "target_rhyme_unit": "N/A", "raw_arpabet": target_pron_raw, "candidates": []}

    target_rhyme_len = len(target_rhyme_unit)
    candidates_list = []
    
    # ----------------------------------------------------------------
    # CMUDict 전체 스캔 로직
    # ----------------------------------------------------------------
    for word, _ in p_dict.items():
        
        candidate_info = get_rhyme_unit(word)
        if not candidate_info:
            continue
            
        candidate_pron_raw, candidate_onset, candidate_rhyme_unit, candidate_vowel = candidate_info
        
        if word == target_word.lower():
            continue
            
        score = 0.0
        rhyme_type = "Slant/Poor Match"
        
        # A. Perfect Rhyme (완벽한 라임) 검사
        # 1. 라임 유닛 길이가 같아야 함
        if len(candidate_rhyme_unit) == target_rhyme_len:
            # 2. 강세 모음부터 끝까지 음소가 완전히 일치해야 함
            if candidate_rhyme_unit == target_rhyme_unit:
                # 3. Onset Check: 강세 모음 앞의 마지막 음소는 달라야 함 (e.g., 'cat'/'sat'은 True Rhyme, 'cat'/'cater'는 아님)
                is_onset_different = (not target_onset or not candidate_onset or target_onset[-1] != candidate_onset[-1])
                
                if is_onset_different:
                    score = 1.0 
                    rhyme_type = "Perfect Rhyme (True Rhyme)"
                # Onset이 같으면 Slant Rhyme 계산으로 넘어감 (예외적인 Near Rhyme)
        
        # B. Slant Rhyme (불완전 라임) 및 Multi-Syllable Rhyme 검사
        if score < 1.0:
            
            # 멀티-음절 라임을 위한 Slant Rhyme 계산
            len_diff = abs(len(candidate_rhyme_unit) - target_rhyme_len)
            
            # 길이 차이가 음소 2개 이하인 경우만 계산
            if len_diff <= 2: 
                
                # 코사인 유사도 계산
                slant_score = calculate_slant_score(target_rhyme_unit, candidate_rhyme_unit)
                
                # 완벽 라임은 아니지만, 음소 유사도가 높을 때
                if slant_score >= 0.85: 
                    score = slant_score
                    rhyme_type = "Multi-Syllable Slant Rhyme"
                elif slant_score >= 0.70:
                    score = slant_score
                    rhyme_type = "Near Rhyme"
                else:
                    continue # 점수가 너무 낮으면 제외
            else:
                continue # 길이가 너무 다르면 제외

        if score > 0.0:
            candidates_list.append({
                "word": word,
                "score": round(score, 4), 
                "ipa": arpabet_to_ipa(candidate_rhyme_unit),
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
# 3. Streamlit UI (수정된 로직 반영)
# =========================================================

st.set_page_config(page_title="Phonetics Analyzer (Advanced Rhyme)", layout="centered")

st.title("🎤 CMUDict 통합: 고급 라임 분석기 (강세 & 슬랜트 반영)")
st.caption("✅ **마지막 강세 모음**부터 라임 유닛을 추출하고, **멀티-음절 슬랜트 라임**을 위한 유사도 점수를 계산합니다.")

# 사용자 입력
input_word = st.text_input("분석할 단어를 입력하세요 (예: magical, mind, computer)", "magical")

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
        st.warning(f"CMUDict에서 '{input_word}'에 대한 적절한 라임 후보를 찾지 못했습니다.")

    # 3. Gemini가 받을 API 응답 (JSON)
    st.markdown("---")
    st.markdown("#### 🤖 Gemini에게 제공할 최종 API 응답 (JSON)")
    final_json_output = {
        "target_word": analysis_result["target_word"],
        "target_ipa": analysis_result["target_ipa"],
        "candidates": [{k: v for k, v in c.items() if k not in ['rhyme_unit']} for c in analysis_result["candidates"]]
    }
    st.code(json.dumps(final_json_output, indent=2), language='json')
