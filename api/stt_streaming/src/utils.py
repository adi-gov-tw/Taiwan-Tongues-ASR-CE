import logging, os, json, ast
from datetime import datetime

# from whisper.utils import get_writer  # 暫時註解掉，因為可能沒有安裝 whisper
from typing import Iterator, TextIO


def srt_format_timestamp(seconds: float):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    return (f"{hours}:") + f"{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def list_to_srt_text(subtitles):
    srt_text = ""
    counter = 1

    for subtitle in subtitles:
        start_time = subtitle["startTime"]
        end_time = subtitle["endTime"]
        text = subtitle["text"]

        srt_text += f"{counter}\n"
        srt_text += (
            f"{srt_format_timestamp(start_time)} --> {srt_format_timestamp(end_time)}\n"
        )
        srt_text += f"{text}\n\n"

        counter += 1

    return srt_text


def list_to_plain_text(subtitles):
    plain_text = ""

    for subtitle in subtitles:
        text = subtitle["text"]
        plain_text += f"{text}\n"

    return plain_text


def convert_transcript_to_subtitles(input_file):
    logging.info(f"convert_transcript_to_subtitles: {input_file}")
    result = None
    if os.path.exists(input_file) and result is None:
        with open(input_file, "r", encoding="utf-8") as f:
            result = f.read()

    if result is None:
        raise Exception("result is empty")

    output_srt_path = os.path.splitext(input_file)[0] + ".srt"
    output_txt_path = os.path.splitext(input_file)[0] + ".txt"
    list = ast.literal_eval(result)
    # 轉換成 SRT 格式
    srt_content = list_to_srt_text(list)
    with open(output_srt_path, "w", encoding="utf-8") as srt_file:
        srt_file.write(srt_content)

    # 轉換成純文字格式
    plain_text_content = list_to_plain_text(list)
    with open(output_txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(plain_text_content)

    duration = list[-1]["endTime"]
    return duration, output_srt_path, output_txt_path


def create_today_folders(directory, job_id):
    logging.info(f"create_today_folders: {directory}, {job_id}")
    current_date = datetime.now()
    year = current_date.year
    month = current_date.month
    day = current_date.day
    folder_path = os.path.join(directory, str(year), str(month), str(day), job_id)
    create_folders(folder_path)
    return folder_path


def create_folders(folder_path):
    logging.info(f"create_folders: {folder_path}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def transfer_streaming_format(list):
    if not list or len(list) == 0:
        return None
    results = []
    for segment in list:
        results.append(
            {
                "startTime": segment["startTime"],
                "endTime": segment["endTime"],
                "text": segment["text"],
                "final": True,
            }
        )
    return results


def transfer_offline_format(data):
    if not data:
        return None
    results = []
    segments = filter_offline_segments(data["segments"])
    for segment in segments:
        logging.info(f"segment: {segment}")
        results.append(
            {
                "startTime": segment["start"],
                "endTime": segment["end"],
                "text": segment["text"],
                "final": True,
            }
        )
    return results


ignore_text = [
    "字幕by",
    "中文字幕由",
    "中文字幕 by",
    "中文字幕提供",
    "請你一定要顯示繁體中文",
    "订阅",
    "打赏",
    "不吝點贊",
    "阿波羅網編譯",
    "逐字稿機器",
    "請看影片資訊欄",
    "Amara.org",
    "整理&字幕志願者",
    "以上言論不代表本台立場",
    "點點欄目",
    "不吝點贊",
    "訂閱轉發",
    "喜歡請訂閱",
    "按讚及分享",
]


def filter_text(text):
    if any(ignore_text_str in text for ignore_text_str in ignore_text):
        return None
    return text


def filter_offline_segments(segments):
    filtered = []
    for segment in segments:
        if any(
            ignore_text_str in segment["text"] in ignore_text_str
            for ignore_text_str in ignore_text
        ):
            logging.warning(f"Segment filtered out due to ignore text: {segment.text}")
            continue
        filtered.append(segment)
    return filtered


language_codes = {
    "afrikaans": "af",
    "amharic": "am",
    "arabic": "ar",
    "assamese": "as",
    "azerbaijani": "az",
    "bashkir": "ba",
    "belarusian": "be",
    "bulgarian": "bg",
    "bengali": "bn",
    "tibetan": "bo",
    "breton": "br",
    "bosnian": "bs",
    "catalan": "ca",
    "czech": "cs",
    "welsh": "cy",
    "danish": "da",
    "german": "de",
    "greek": "el",
    "english": "en",
    "spanish": "es",
    "estonian": "et",
    "basque": "eu",
    "persian": "fa",
    "finnish": "fi",
    "faroese": "fo",
    "french": "fr",
    "galician": "gl",
    "gujarati": "gu",
    "hausa": "ha",
    "hawaiian": "haw",
    "hebrew": "he",
    "hindi": "hi",
    "croatian": "hr",
    "haitian": "ht",
    "hungarian": "hu",
    "armenian": "hy",
    "indonesian": "id",
    "icelandic": "is",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jw",
    "georgian": "ka",
    "kazakh": "kk",
    "khmer": "km",
    "kannada": "kn",
    "korean": "ko",
    "latin": "la",
    "luxembourgish": "lb",
    "lingala": "ln",
    "lao": "lo",
    "lithuanian": "lt",
    "latvian": "lv",
    "malagasy": "mg",
    "maori": "mi",
    "macedonian": "mk",
    "malayalam": "ml",
    "mongolian": "mn",
    "marathi": "mr",
    "malay": "ms",
    "maltese": "mt",
    "burmese": "my",
    "nepali": "ne",
    "dutch": "nl",
    "norwegian nynorsk": "nn",
    "norwegian": "no",
    "occitan": "oc",
    "punjabi": "pa",
    "polish": "pl",
    "pashto": "ps",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "sanskrit": "sa",
    "sindhi": "sd",
    "sinhalese": "si",
    "slovak": "sk",
    "slovenian": "sl",
    "shona": "sn",
    "somali": "so",
    "albanian": "sq",
    "serbian": "sr",
    "sundanese": "su",
    "swedish": "sv",
    "swahili": "sw",
    "tamil": "ta",
    "telugu": "te",
    "tajik": "tg",
    "thai": "th",
    "turkmen": "tk",
    "tagalog": "tl",
    "turkish": "tr",
    "tatar": "tt",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
    "vietnamese": "vi",
    "yiddish": "yi",
    "yoruba": "yo",
    "simplifiedchinese": "zh",
    "traditionalchinese": "zh",
    "cantonese": "yue",
}


# 測試代碼
class Segment:
    def __init__(self, text):
        self.text = text


if __name__ == "__main__":
    """
    segments = [
        "這是一段繁體中文。",
        "这是一段简体中文。",
        "This is a test.",
        "忽略這個文本",
        "另一段繁體中文。",
        "請不吝點賞 订阅 分享 打赏支持明镜与点点栏目",
        "我在阅读《简爱》这本书。",
        "【阿波羅網編譯】",
        "喜歡請訂閱按讚及分享",
    ]

    for text in segments:
        if filter_text(text) is None:
            print(f"Filtered out: {text}")
    """
    json_file = "/home/asr/subtitles/2024/2/20/7e329658ad0e40e4b23e350937c72b84/20240220040033.json"
    duration, output_srt_path, output_txt_path = convert_transcript_to_subtitles(
        json_file
    )
    print(
        f"duration: {duration}, output_srt_path: {output_srt_path}, output_txt_path: {output_txt_path}"
    )
