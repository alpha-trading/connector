# -*- coding:utf-8 -*-
from datetime import date
import dart_fss as dart
import re
from bs4 import BeautifulSoup
from typing import Optional


class Dartfss:
    def get_report_values(self, table, report_date: str, is_consolidated: bool):
        """
        :param table: 당기실적표
        :param report_date: 보고서 게시날짜
        :param is_consolidated: 연결제무재표일 경우 True, 아닐 경우 False
        :return: 당기실적 dict { 매출액, 영업이익, 법인세비용차감전 계속사업이익, 당기순이익, 지배기업 소유주지분 순이익 }
        """
        money = table[0].text
        if money == '단위 : 조원, %':
            money = 1000000
        elif money == '단위 : 억원, %':
            money = 100
        else:
            money = 1
        # 매출액
        sales = re.sub('[^0-9]', '', table[4].text)
        sales = int(sales) * money if sales != '' else None
        # 영업이익
        operating_income = re.sub('[^0-9]', '', table[14].text)
        operating_income = int(operating_income) * money if operating_income != '' else None
        # 법인세비용차감전 계속사업이익
        income_from_continuing_operations_before_tax = re.sub('[^0-9]', '', table[24].text)
        income_from_continuing_operations_before_tax = int(income_from_continuing_operations_before_tax) * money \
            if income_from_continuing_operations_before_tax != '' else None
        # 당기순이익
        net_income = re.sub('[^0-9]', '', table[34].text)
        net_income = int(net_income) * money if net_income != '' else None
        # 지배기업 소유주지분 순이익, 연결제무재표에만 추가되는 값
        if is_consolidated:
            controlling_interest = re.sub('[^0-9]', '', table[44].text)
            controlling_interest = int(controlling_interest) * money if controlling_interest != '' else None
        else:
            controlling_interest = None
    
        data = {'date': report_date, 'sales': sales, 'operating_income': operating_income,
                'income_from_continuing_operations_before_tax': income_from_continuing_operations_before_tax,
                'net_income': net_income, 'controlling_interest': controlling_interest}
        return data
    
    def get_expected_sales_report(self, api_key: str, corp_code: str, start_date: Optional[date] = None, end_date: Optional[date] = None):
        """
        :param api_key: dart api key
        :param corp_code: corp_code
        :param start_date: search report start date
        :param end_date: search report end date
        :return: consolidated_expected_report, revised_consolidated_expected_report, expected_report, revised_expected_report
        연결재무제표기준영업(잠정)실적(공정공시), [기재정정]연결재무제표기준영업실적등에대한전망(공정공시),
        영업(잠정)실적(공정공시), [기재정정]영업(잠정)실적(공정공시)
        """
        # 증권 고유번호를 이용해 정보검색
        dart.set_api_key(api_key=api_key)
        corp_list = dart.get_corp_list()
        corp = corp_list.find_by_corp_code(corp_code=corp_code)
    
        consolidated_expected_report = []
        expected_report = []
        revised_consolidated_expected_report = []
        revised_expected_report = []
    
        pages = corp.search_filings(bgn_de=start_date, end_de=end_date, pblntf_detail_ty=['I002']).total_page + 1
        for page in range(1, pages):
            reports = corp.search_filings(bgn_de=start_date, end_de=end_date, pblntf_detail_ty=['I002'], page_no=page)
            for i in reports:
                report_date = i.info['rcept_dt']
                report_name = i.info['report_nm']
                source = BeautifulSoup(i.pages[0].html, 'html.parser')
    
                if report_name == '연결재무제표기준영업(잠정)실적(공정공시)':
                    table = source.find_all("span", {"class": "xforms_input"})
                    data = self.get_report_values(table, report_date, True)
                    consolidated_expected_report.append(data)
    
                elif report_name == '[기재정정]연결재무제표기준영업(잠정)실적(공정공시)':
                    table = source.select("table")[-1].find_all("span", {"class": "xforms_input"})
                    data = self.get_report_values(table, report_date, True)
                    revised_consolidated_expected_report.append(data)
    
                elif report_name == '영업(잠정)실적(공정공시)':
                    table = source.find_all("span", {"class": "xforms_input"})
                    data = self.get_report_values(table, report_date, False)
                    expected_report.append(data)
    
                elif report_name == '[기재정정]영업(잠정)실적(공정공시)':
                    table = source.select("table")[-1].find_all("span", {"class": "xforms_input"})
                    data = self.get_report_values(table, report_date, False)
                    revised_expected_report.append(data)
    
        return consolidated_expected_report, revised_consolidated_expected_report, expected_report, revised_expected_report