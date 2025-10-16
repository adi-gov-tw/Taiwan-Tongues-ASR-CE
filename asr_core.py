import os
from datetime import datetime, timedelta
from faster_whisper import WhisperModel
import time
from pathlib import Path
import librosa
import soundfile as sf
import pywer
import re
import cn2an
import pandas as pd
import opencc
import unicodedata
import argparse
import glob
import json
from cer import compare_texts

s2tw = opencc.OpenCC("s2tw")


def split_sentence_to_words(text: str, is_split: bool):
    if is_split is False:
        return text
    pattern = re.compile(
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF%]|\d+\.\d+|\d+)"
    )
    chars = pattern.split(text.strip().lower())
    return " ".join([w.strip() for w in chars if w is not None and w.strip()])


def replace_words(article):
    mappings = {
        "百分之十五": "15%",
        "百分之五": "5%",
        "百分之十二點五": "12.5%",
        "百分之七": "7%",
        "零八零零零九五九八": "080009598",
    }
    replaced_article = article
    for old, new in mappings.items():
        replaced_article = replaced_article.replace(old, new)
    return replaced_article


def convert_time(time):
    time_str = f"{time:.3f}"
    if "." in time_str:
        seconds, millisecond = time_str.split(".")
    else:
        time = time_str
        millisecond = "000"

    delta = timedelta(seconds=int(seconds))
    time_str = (datetime.min + delta).strftime("%H:%M:%S")

    t = str(time_str).split(":")
    return f"{':'.join([x.zfill(2) for x in t])}.{millisecond}"


def full_to_half(text):
    half_width_text = ""
    for char in text:
        half_char = unicodedata.normalize("NFKC", char)
        if half_char.isalpha():
            half_char = half_char
        half_width_text += half_char
    return half_width_text


def remove_special_characters_by_dataset_name(text):
    # 移除特殊字符
    chars_to_ignore_regex_base = r'[,"\'。，^¿¡；「」《》:：＄$\[\]〜～·・‧―─–－⋯、＼【】=<>{}_〈〉　）（—『』«»→„…(),`&＆﹁﹂#＃\\!?！;]'

    sentence = re.sub(chars_to_ignore_regex_base, "", text)
    sentence = full_to_half(sentence)

    return sentence


def num_to_cn(text, mode=0):
    method = "an2cn" if mode == 0 else "cn2an"
    text = cn2an.transform(text, method)
    return text


def find_original_transcript(audio_file):
    """尋找對應的原始逐字稿檔案"""
    audio_dir = os.path.dirname(audio_file)
    audio_name = os.path.splitext(os.path.basename(audio_file))[0]

    # 可能的原始逐字稿檔案名稱模式
    possible_names = [
        f"{audio_name}.txt",
        f"{audio_name}_transcript.txt",
        f"{audio_name}_original.txt",
        f"{audio_name}_reference.txt",
        f"{audio_name}_ground_truth.txt",
    ]

    for name in possible_names:
        transcript_path = os.path.join(audio_dir, name)
        if os.path.exists(transcript_path):
            return transcript_path

    return None


def process_audio_folder(folder_path, output_file="transcription_results.txt"):
    """
    處理指定資料夾中的所有音檔

    Args:
        folder_path: 音檔資料夾路徑
        output_file: 輸出檔案名稱 (已棄用，保留用於向後相容)
    """
    # 支援的音檔格式
    audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.aac"]

    # 取得所有音檔
    audio_files = []
    for ext in audio_extensions:
        # 使用 case-insensitive 搜尋，避免重複計算
        found_files = glob.glob(os.path.join(folder_path, ext), recursive=False)
        found_files.extend(
            glob.glob(os.path.join(folder_path, ext.upper()), recursive=False)
        )
        audio_files.extend(found_files)

    # 移除重複的檔案（因為大小寫搜尋可能找到同一個檔案）
    audio_files = list(set(audio_files))

    if not audio_files:
        print(f"在資料夾 {folder_path} 中找不到音檔")
        return

    print(f"找到 {len(audio_files)} 個音檔")

    # 載入模型
    try:
        model = WhisperModel("models", device="cuda", compute_type="float16")
        print("模型載入成功: models")
    except Exception as e:
        print(f"模型載入失敗: {e}")
        return

    # 儲存所有比對結果
    comparison_results = []

    # 處理每個音檔
    for i, audio_file in enumerate(audio_files, 1):
        print(f"處理音檔 {i}/{len(audio_files)}: {os.path.basename(audio_file)}")

        try:
            # 載入音檔
            audio, sr = librosa.load(audio_file, sr=16000, mono=False)

            # 轉錄
            segments, info = model.transcribe(
                audio,
                language="zh",
                word_timestamps=False,
                vad_filter=True,
                beam_size=5,
                condition_on_previous_text=True,
                initial_prompt="",
            )

            # 組合轉錄結果
            text = ""
            for segment in segments:
                text += segment.text

            # 後處理
            processed_text = remove_special_characters_by_dataset_name(
                s2tw.convert(replace_words(text))
            ).lower()

            # 生成輸出檔案路徑
            audio_dir = os.path.dirname(audio_file)
            audio_name = os.path.splitext(os.path.basename(audio_file))[0]
            output_path = os.path.join(audio_dir, f"{audio_name}_asr.txt")

            # 儲存轉錄結果
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"{processed_text}")

            print(f"轉錄結果已儲存至: {output_path}")
            print(f"轉錄結果: {processed_text}")

            # 尋找並比對原始逐字稿
            original_transcript_path = find_original_transcript(audio_file)
            comparison_result = {
                "audio_file": os.path.basename(audio_file),
                "asr_result": processed_text,
                "original_transcript": None,
                "cer_result": None,
                "has_original_transcript": False,
            }

            if original_transcript_path:
                try:
                    with open(original_transcript_path, "r", encoding="utf-8") as f:
                        original_text = f.read().strip()

                    comparison_result["original_transcript"] = original_text
                    comparison_result["has_original_transcript"] = True

                    # 進行 CER 比對
                    cer_result = compare_texts(original_text, processed_text)
                    if cer_result:
                        comparison_result["cer_result"] = {
                            "correct_rate": cer_result.correct_rate,
                            "cer_rate": cer_result.cer_rate,
                            "total_errors": cer_result.total_errors,
                            "substitutions_count": cer_result.substitutions_count,
                            "deletions_count": cer_result.deletions_count,
                            "insertions_count": cer_result.insertions_count,
                            "total_chars": cer_result.total_chars,
                            "substitutions_errors": cer_result.substitutions_errors,
                            "deletions_errors": cer_result.deletions_errors,
                            "insertions_errors": cer_result.insertions_errors,
                            "reference_highlighted": cer_result.reference_highlighted,
                            "hypothesis_highlighted": cer_result.hypothesis_highlighted,
                        }

                        print(f"原始逐字稿: {original_text}")
                        print(
                            f"CER: {cer_result.cer_rate:.4f}, 正確率: {cer_result.correct_rate:.2f}%"
                        )
                        print(
                            f"替換錯誤: {cer_result.substitutions_count}, 刪除錯誤: {cer_result.deletions_count}, 插入錯誤: {cer_result.insertions_count}"
                        )
                    else:
                        print("CER 比對失敗")

                except Exception as e:
                    print(f"讀取原始逐字稿時發生錯誤: {e}")
            else:
                print(f"找不到對應的原始逐字稿檔案")

            comparison_results.append(comparison_result)

        except Exception as e:
            print(f"處理音檔 {audio_file} 時發生錯誤: {e}")
            # 即使發生錯誤也建立輸出檔案
            audio_dir = os.path.dirname(audio_file)
            audio_name = os.path.splitext(os.path.basename(audio_file))[0]
            output_path = os.path.join(audio_dir, f"{audio_name}_asr.txt")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"檔案名稱: {os.path.basename(audio_file)}\n")
                f.write(f"錯誤: {e}\n")

            print(f"錯誤記錄已儲存至: {output_path}")

            # 添加錯誤結果到比較結果中
            comparison_results.append(
                {
                    "audio_file": os.path.basename(audio_file),
                    "asr_result": None,
                    "original_transcript": None,
                    "cer_result": None,
                    "has_original_transcript": False,
                    "error": str(e),
                }
            )

    # 計算整體統計
    total_files = len(comparison_results)
    files_with_transcript = sum(
        1 for r in comparison_results if r.get("has_original_transcript", False)
    )
    files_with_cer = sum(
        1 for r in comparison_results if r.get("cer_result") is not None
    )

    if files_with_cer > 0:
        avg_cer = (
            sum(
                r["cer_result"]["cer_rate"]
                for r in comparison_results
                if r.get("cer_result")
            )
            / files_with_cer
        )
        avg_correct_rate = (
            sum(
                r["cer_result"]["correct_rate"]
                for r in comparison_results
                if r.get("cer_result")
            )
            / files_with_cer
        )
        total_substitutions = sum(
            r["cer_result"]["substitutions_count"]
            for r in comparison_results
            if r.get("cer_result")
        )
        total_deletions = sum(
            r["cer_result"]["deletions_count"]
            for r in comparison_results
            if r.get("cer_result")
        )
        total_insertions = sum(
            r["cer_result"]["insertions_count"]
            for r in comparison_results
            if r.get("cer_result")
        )
    else:
        avg_cer = 0
        avg_correct_rate = 0
        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0

    # 建立最終結果
    final_result = {
        "summary": {
            "total_files": total_files,
            "files_with_transcript": files_with_transcript,
            "files_with_cer": files_with_cer,
            "average_cer": avg_cer,
            "average_correct_rate": avg_correct_rate,
            "total_substitutions": total_substitutions,
            "total_deletions": total_deletions,
            "total_insertions": total_insertions,
        },
        "detailed_results": comparison_results,
    }

    # 輸出 JSON 到根目錄
    output_json_path = os.path.join(os.getcwd(), "asr_comparison_results.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print(f"\n=== 處理完成 ===")
    print(f"總檔案數: {total_files}")
    print(f"有原始逐字稿的檔案數: {files_with_transcript}")
    print(f"成功比對的檔案數: {files_with_cer}")
    if files_with_cer > 0:
        print(f"平均 CER: {avg_cer:.4f}")
        print(f"平均正確率: {avg_correct_rate:.2f}%")
        print(f"總替換錯誤: {total_substitutions}")
        print(f"總刪除錯誤: {total_deletions}")
        print(f"總插入錯誤: {total_insertions}")
    print(f"詳細結果已儲存至: {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description="音檔轉錄工具")
    parser.add_argument("folder", help="音檔資料夾路徑")
    parser.add_argument(
        "--output",
        default="transcription_results.txt",
        help="輸出檔案名稱 (已棄用，保留用於向後相容)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.folder):
        print(f"資料夾不存在: {args.folder}")
        return

    process_audio_folder(args.folder, args.output)


if __name__ == "__main__":
    main()
