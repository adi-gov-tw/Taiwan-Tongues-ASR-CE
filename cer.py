import difflib
import re
import html

# 同音字或特定替換詞的強制轉換表
# 鍵是原始字，值是目標字
HOMOPHONE_MAPPING = {
    "她": "他",
    "它": "他",
    "臺": "台",
    "著": "著",
    "的": "的",
    "得": "的",
    # 請在這裡加入更多需要強制轉換的字詞
}


class CERResult:
    """CER 比對結果物件"""

    def __init__(self, reference_text, hypothesis_text):
        self.reference_text = reference_text
        self.hypothesis_text = hypothesis_text
        self.reference_cleaned = ""
        self.hypothesis_cleaned = ""
        self.correct_rate = 0.0
        self.cer_rate = 0.0
        self.total_errors = 0
        self.substitutions_count = 0
        self.deletions_count = 0
        self.insertions_count = 0
        self.total_chars = 0
        self.substitutions_errors = []
        self.deletions_errors = []
        self.insertions_errors = []
        self.reference_highlighted = ""
        self.hypothesis_highlighted = ""


# 中英文數字轉換處理
def arabic_to_chinese_number(num_str):
    chinese_numerals = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]

    # 如果以 '0' 開頭且長度大於1，或者長度超過9，就當作「數字序列」
    if (num_str.startswith("0") and len(num_str) > 1) or len(num_str) > 9:
        # 逐字轉換
        return "".join([chinese_numerals[int(digit)] for digit in num_str])

    units = ["", "十", "百", "千", "萬", "十萬", "百萬", "千萬", "億"]
    result = []
    length = len(num_str)
    zero_flag = False

    # 確保 num_str 是一個整數，以避免前導零的問題
    try:
        num = int(num_str)
        num_str = str(num)  # 重新生成不帶前導零的字串
        length = len(num_str)
    except ValueError:
        return ""  # 如果轉換失敗，返回空字串

    for i, digit in enumerate(num_str):
        n = int(digit)
        unit = units[length - i - 1]
        if n == 0:
            if not result or (result and result[-1] == chinese_numerals[0]):
                zero_flag = True
            else:
                zero_flag = True
        else:
            if zero_flag:
                result.append(chinese_numerals[0])
                zero_flag = False
            result.append(chinese_numerals[n] + unit)

    if result and result[-1] == chinese_numerals[0] and len(result) > 1:
        result.pop()

    if not result:
        return chinese_numerals[0]

    if len(result) == 2 and result[0] == "一十":
        result[0] = "十"

    return "".join(result)


# 清理文本、數字
def clean_text(text, to_lower=True):
    # 將所有換行符（\n）替換為
    text_processed = text.replace("\n", " ").replace("\r", "")

    # 執行同音字或其他指定字詞的強制轉換
    # 這必須在移除標點和空格之前進行，以確保原始字詞能被匹配到
    for old_char, new_char in HOMOPHONE_MAPPING.items():
        text_processed = text_processed.replace(old_char, new_char)

    # 數字轉換為中文數字 (在移除空格和標點之前進行，以確保數字本身是連續的)
    temp_text_with_chinese_numbers = ""
    last_idx = 0
    # 找到所有連續的數字序列
    for m in re.finditer(r"\d+", text_processed):
        # 添加數字之前的部分
        temp_text_with_chinese_numbers += text_processed[last_idx : m.start()]
        # 添加轉換後的中文數字
        temp_text_with_chinese_numbers += arabic_to_chinese_number(m.group(0))
        last_idx = m.end()
    # 添加數字之後的剩餘部分
    temp_text_with_chinese_numbers += text_processed[last_idx:]
    text_processed = temp_text_with_chinese_numbers
    cleaned_text = re.sub(r"[^\w]", "", text_processed).replace(" ", "")
    cleaned_text = re.sub(
        r"[^\u4e00-\u9fa5a-zA-Z]", "", text_processed
    )  # 移除所有非中文和非英文字符
    cleaned_text = cleaned_text.replace(" ", "")
    final_text = text.replace("\n", "").replace("\r", "")  # 移除換行符

    # 執行同音字替換
    for old_char, new_char in HOMOPHONE_MAPPING.items():
        final_text = final_text.replace(old_char, new_char)

    # 數字轉換
    temp_final_text = ""
    last_idx = 0
    for m in re.finditer(r"\d+", final_text):
        temp_final_text += final_text[last_idx : m.start()]
        temp_final_text += arabic_to_chinese_number(m.group(0))
        last_idx = m.end()
    temp_final_text += final_text[last_idx:]
    final_text = temp_final_text

    # 移除所有非漢字、非英文字母的字符，並移除所有空格
    # \u4e00-\u9fa5 是中文字的 Unicode 範圍
    # a-zA-Z 是英文字母
    cleaned_text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", "", final_text)

    return cleaned_text.lower() if to_lower else cleaned_text.lower()


# 計算 CER 並逐字處理顏色
def calculate_cer(reference, hypothesis):
    # 建立結果物件
    result = CERResult(reference, hypothesis)

    # 先清理文本，應用同音字替換和數字轉換
    result.reference_cleaned = clean_text(reference)
    result.hypothesis_cleaned = clean_text(hypothesis)

    sm = difflib.SequenceMatcher(
        None, result.reference_cleaned, result.hypothesis_cleaned
    )

    result_reference, result_hypothesis = "", ""
    substitutions_count = 0
    insertions_count = 0
    deletions_count = 0
    N = len(result.reference_cleaned)  # N 是清理後的參考文本長度

    break_interval = 250
    char_count = 0

    substitutions_errors = []
    deletions_errors = []
    insertions_errors = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":  # 替換
            ref_substr = result.reference_cleaned[i1:i2]
            hyp_substr = result.hypothesis_cleaned[j1:j2]

            # 替換錯誤的字元數是兩個子字串長度中較小的那個
            substitutions_to_add = min(len(ref_substr), len(hyp_substr))
            substitutions_count += substitutions_to_add

            substitutions_errors.append(
                f"正確文本中的「{ref_substr}」 在 ASR 轉譯文本中被替換成 「{hyp_substr}」"
            )

            if len(hyp_substr) > len(ref_substr):
                insertions_to_add_in_replace = len(hyp_substr) - len(ref_substr)
                insertions_count += insertions_to_add_in_replace
                insertions_errors.append(
                    f"「{hyp_substr[len(ref_substr):]}」 在 ASR 結果 額外輸出，不屬於正確文本內容 (替換造成)"
                )
            elif len(ref_substr) > len(hyp_substr):
                deletions_to_add_in_replace = len(ref_substr) - len(hyp_substr)
                deletions_count += deletions_to_add_in_replace
                deletions_errors.append(
                    f"正確文本中的「{ref_substr[len(hyp_substr):]}」 被刪除，未被 ASR 轉譯成功 (替換造成)"
                )

            # 使用純文字標記而不是 HTML
            result_reference += "".join(
                [
                    f"[{result.reference_cleaned[k]}]"  # 替換錯誤用方括號標記
                    for k in range(i1, i1 + substitutions_to_add)
                ]
            )
            result_hypothesis += "".join(
                [
                    f"[{result.hypothesis_cleaned[k]}]"  # 替換錯誤用方括號標記
                    for k in range(j1, j1 + substitutions_to_add)
                ]
            )

            if len(ref_substr) > len(hyp_substr):
                result_reference += "".join(
                    [
                        f"<{result.reference_cleaned[k]}>"  # 刪除錯誤用尖括號標記
                        for k in range(i1 + substitutions_to_add, i2)
                    ]
                )
                result_hypothesis += "".join(
                    [
                        "□"  # 刪除錯誤用方框標記
                        for _ in range(i1 + substitutions_to_add, i2)
                    ]
                )
            if len(hyp_substr) > len(ref_substr):
                result_hypothesis += "".join(
                    [
                        f"({result.hypothesis_cleaned[k]})"  # 插入錯誤用圓括號標記
                        for k in range(j1 + substitutions_to_add, j2)
                    ]
                )
                result_reference += "".join(
                    [
                        "□"  # 插入錯誤用方框標記
                        for _ in range(j1 + substitutions_to_add, j2)
                    ]
                )

        elif tag == "delete":
            deletions_to_add = i2 - i1
            deletions_count += deletions_to_add
            deletions_errors.append(
                f"正確文本中的「{result.reference_cleaned[i1:i2]}」 被刪除 ，未被 ASR 轉譯成功"
            )
            result_reference += "".join(
                [
                    f"<{result.reference_cleaned[k]}>"  # 刪除錯誤用尖括號標記
                    for k in range(i1, i2)
                ]
            )
            result_hypothesis += "".join(
                ["□" for _ in range(i1, i2)]  # 刪除錯誤用方框標記
            )

        elif tag == "insert":
            insertions_to_add = j2 - j1
            insertions_count += insertions_to_add
            insertions_errors.append(
                f"「{result.hypothesis_cleaned[j1:j2]}」 在 ASR 結果 額外輸出，不屬於正確文本內容"
            )
            result_reference += "".join(
                ["□" for _ in range(j1, j2)]  # 插入錯誤用方框標記
            )
            result_hypothesis += "".join(
                [
                    f"({result.hypothesis_cleaned[k]})"  # 插入錯誤用圓括號標記
                    for k in range(j1, j2)
                ]
            )

        elif tag == "equal":
            result_reference += "".join(
                [result.reference_cleaned[k] for k in range(i1, i2)]
            )
            result_hypothesis += "".join(
                [result.hypothesis_cleaned[k] for k in range(j1, j2)]
            )

        char_count += (i2 - i1) + (j2 - j1)
        if char_count >= break_interval:
            result_reference += "\n\n"
            result_hypothesis += "\n\n"
            char_count = 0

    errors = substitutions_count + deletions_count + insertions_count
    cer = errors / N if N > 0 else 0
    correct_rate = 100 * (1 - cer)

    # 設定結果物件的屬性
    result.correct_rate = correct_rate
    result.cer_rate = cer
    result.total_errors = errors
    result.substitutions_count = substitutions_count
    result.deletions_count = deletions_count
    result.insertions_count = insertions_count
    result.total_chars = N
    result.substitutions_errors = substitutions_errors
    result.deletions_errors = deletions_errors
    result.insertions_errors = insertions_errors
    result.reference_highlighted = result_reference
    result.hypothesis_highlighted = result_hypothesis

    return result


# 簡單的文字比對函數
def compare_texts(reference_text, hypothesis_text):
    """
    比較兩段文字並返回 CER 分析結果物件

    Args:
        reference_text (str): 參考正確文本
        hypothesis_text (str): ASR 轉譯文本

    Returns:
        CERResult: 包含所有比對結果的物件
    """
    if not reference_text or not hypothesis_text:
        return None

    return calculate_cer(reference_text, hypothesis_text)


# 測試用主函數
if __name__ == "__main__":
    # 測試範例
    reference = "今天天氣很好，我們去公園散步。"
    hypothesis = "今天天氣很好!，我去公園散步。"

    result = compare_texts(reference, hypothesis)
    if result:
        print(f"正確率: {result.correct_rate:.2f}%")
        print(f"CER: {result.cer_rate:.4f}")
        print(f"總錯誤數: {result.total_errors}")
        print(f"替換錯誤: {result.substitutions_count}")
        print(f"刪除錯誤: {result.deletions_count}")
        print(f"插入錯誤: {result.insertions_count}")
    else:
        print("請提供兩段文字進行比對。")
