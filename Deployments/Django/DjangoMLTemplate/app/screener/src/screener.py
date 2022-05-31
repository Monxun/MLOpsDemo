import requests 
import pandas as pd

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}


def get_screener(version):
    screen = requests.get(f'https://finviz.com/screener.ashx?v={version}&f=ind_consumerelectronics,sec_technology', headers = headers).text

    tables = pd.read_html(screen)
    tables = tables[-2]
    tables.columns = tables.iloc[0]
    tables = tables[1:]

    return tables


# TURN INTO METHOD AND PASS ARRAY OF TABLE VERSIONS
if __name__ == '__main__':
    tables = {}
    tables['111'] = get_screener('111')
    tables['161'] = get_screener('161')
    tables['121'] = get_screener('121')



    consolidatedtables = pd.merge(tables['111'],tables['161'],how='outer',left_on='Ticker',right_on='Ticker')
    consolidatedtables = pd.merge(consolidatedtables,tables['121'],how='outer',left_on='Ticker',right_on='Ticker')

    consolidatedtables.to_csv('test.csv')

    print(consolidatedtables)