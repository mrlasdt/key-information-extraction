# from pathlib import Path  # add Fiintrade path to import config, required to run main()
# import sys
# FILE = Path(__file__).absolute()
# sys.path.append(FILE.parents[2].as_posix())  # add Fiintrade/ to path
# %%
import re
from datetime import datetime
from sklearn.metrics import classification_report
from src.tools.utils import read_json
from underthesea import word_tokenize

# %%


class DatetimeCorrector:
    @staticmethod
    def verify_and_convert_date(date_str):
        # Try to parse the date string using the datetime module
        try:
            date = datetime.strptime(date_str, "%d/%m/%Y")  # TODO: fix this
        except ValueError:
            # If the date string is not in a valTid format, return False
            return ""

        # If the date string is in the correct format, check if it is already in the "dd/mm/yyyy" format
        if date_str[:2] == "dd" and date_str[3:5] == "mm" and date_str[6:] == "yyyy":
            # If the date string is already in the correct format, return it as is
            return date_str
        else:
            # If the date string is not in the correct format, use the strftime method to convert it
            return date.strftime("%d/%m/%Y")

    @staticmethod
    def get_date_from_date_string_by_prefix(date_string_, prefix_):
        prefix = prefix_.lower()
        date_string = date_string_.lower()
        if prefix in date_string:
            try:
                if prefix == "năm":
                    match = re.split(
                        r"năm[^\d]*(\d{4}|\d{1}[\s.]*\d{3}|\d{3}[\s.]*\d{1}|\d{2}[\s.]*\d{2}|\d{2}[\s.]*\d{1}[\s.]*\d{1}|\d{1}[\s.]*\d{2}[\s.]*\d{1}|\d{1}[\s.]*\d{1}[\s.]*\d{2}|\d{1}[\s.]*\d{1}[\s.]*\d{1}[\s.]*\d{1})[\s.]*\b",
                        date_string)  # match "năm" following with all combination of 4 numbers and whitespace/dot such as 1111; 111.1; 111 1; 11 2 1, 2 2 2.2; ...
                elif prefix == "ngày":
                    match = re.split(r"ngày[^\d]*(\d{2}|\d{1}[\s.]*\d{1}|\d{1})[\s.]*\b", date_string)
                else:
                    match = re.split(r"tháng[^\d]*(\d{2}|\d{1}[\s.]*\d{1}|\d{1})[\s.]*\b", date_string)
                num = match[1]
                remain_string = match[2] if prefix != "năm" else match[0]
                return num, remain_string
            except:
                return "", date_string_
        else:
            return '', date_string_

    @staticmethod
    def get_date_by_pattern(date_string):
        match = re.findall(r"([^\d\s]+)?\s*(\d{1}\s*\d?\s+|\d{2}\s+|\d+\s*\b)", date_string)
        if not match:
            return ""
        if len(match) > 3:
            day = match[0][-1].replace(" ", "")
            year = match[-1][-1].replace(" ", "")
            # since in the VIETNAMESE DRIVER LICENSE, the tháng/month is behind the stamp and can be recognized as any thing => mistạken number may be in range (1->-3) => choose month to be -2
            month = match[-2][-1].replace(" ", "")
            return "/".join([day, month, year])
        else:
            return "/".join([m[-1].replace(" ", "") for m in match])

    @staticmethod
    def extract_date_from_string(date_string):
        remain_str = date_string
        ldate = []
        for d in ["năm", "ngày", "tháng"]:
            # date, remain_str = get_date_from_date_string_by_prefix(remain_str, d)
            date, remain_str = DatetimeCorrector.get_date_from_date_string_by_prefix(date_string, d)
            if not date:
                return DatetimeCorrector.get_date_by_pattern(date_string)
            ldate.append(date.strip().replace(" ", "").replace(".", ""))
        return "/".join([ldate[1], ldate[2], ldate[0]])

    @staticmethod
    def correct(date_string):
        # Extract the day, month, and year from the string using regular expressions
        date_string = date_string.lower().replace("✪", " ")
        date_string = " ".join(word_tokenize(date_string))
        parsed_date_string_ = DatetimeCorrector.verify_and_convert_date(date_string)  # if already in datetime format
        if parsed_date_string_:
            return parsed_date_string_
        # match = re.findall(r"([^\d\s]+)?\s*(\d{1}\s*\d?\s+|\d{2}\s+|\d{2,4}\b|\d{3}\b)", date_string)
        extracted_date = DatetimeCorrector.extract_date_from_string(date_string)
        parsed_date_string_ = DatetimeCorrector.verify_and_convert_date(extracted_date)
        # print(extracted_date, parsed_date_string_)
        return parsed_date_string_ if parsed_date_string_ else date_string

    @staticmethod
    def eval():
        data = read_json("data/dates_gplx.json")
        # data = read_json("data/dates_invoice.json")
        type_column = "GPLX"  # Invoice/GPLX
        y_true, y_pred = [], []
        # lexcludes = {"03.pe_0087_000", "03.pe_1420_001", "135", "20210_1099_000", "20210_1140_000", "330", "360",
        #              "hoa_d_0195_000", "invoice_0864_000", "invoice_1392_000", "invoice_195", "mngtf_0244_000", "invoice_0586_000"}
        lexcludes = {}
        ddata = {}
        for k, d in data.items():
            if k in lexcludes:
                continue
            if k == "inv_SDV_215":
                print("debugging")
            pred = DatetimeCorrector.correct(d["pred"])
            label = DatetimeCorrector.correct(d["label"])
            ddata[k] = {}
            data[k]["Type"] = type_column
            ddata[k]["Predict"] = d["pred"]
            ddata[k]["Label"] = d["label"]
            ddata[k]["Post-processed"] = pred
            y_pred.append(pred == label)
            y_true.append(1)
            # if pred != label:
            if k == "invoice_1219_000":
                print("\n", k, '-' * 50)
                print(pred, "------", d["pred"])
                print(label, "------", d["label"])
        print(classification_report(y_true, y_pred))
        import pandas as pd
        df = pd.DataFrame.from_dict(ddata, orient="index")
        df.to_csv(f"result/datetime_post_processed_{type_column}.csv")


# %%
if __name__ == "__main__":
    # %%
    print(DatetimeCorrector.correct("ngày /date 01 tháng /month 04 năm/year✪2022"))

# %%
