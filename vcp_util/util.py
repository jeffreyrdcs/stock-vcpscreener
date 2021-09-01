'''
Utility library for VCP screener
'''

import os
import numpy as np
import pandas as pd
import img2pdf
from fpdf import FPDF
from PyPDF2 import PdfFileMerger, PdfFileReader
from PIL import Image
import subprocess
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
from datetime import datetime


def get_last_trade_day():
    '''
    Get last trade day using current UTC time
    '''
    curr_day = datetime.utcnow() - timedelta(hours=5)       # UTC -5, i.e. US NY timezone

    # If sat or sun, return the closest weekday (i.e. friday)
    if curr_day.isoweekday() in set((6, 7)):
        trade_day = curr_day - timedelta(curr_day.isoweekday() % 5)
    else:
        trade_day = curr_day
    return trade_day


def convert_strdate_datetime(in_strdate):
    '''
    Convert the date from a format of YYYY-MM-DD to a datetime object
    '''
    return datetime.strptime(in_strdate, '%Y-%m-%d')


def convert_png_jpg(in_pngname, out_jpgname):
    '''
    Convert PNG to JPG with PIL
    '''
    im = Image.open(in_pngname)
    im = im.convert("RGB")
    im.save(out_jpgname)


def cleanup_dir_jpg_png(input_dir):
    '''
    Clean up an input directory, delete all pngs and jpgs
    '''
    outlist = os.listdir(input_dir)
    for item in outlist:
        if item.endswith(".png"):
            os.remove(os.path.join(input_dir, item))
        if item.endswith(".jpg"):
            os.remove(os.path.join(input_dir, item))


def gen_report_front_page(in_dict, instockname, output_dir):
    '''
    Print a stock report to a PDF
    Write it to an output directory
    '''
    out_msg = []
    instockname.sort()

    total = in_dict['adv'] + in_dict['decl']
    out_msg.append(("Advance / Decline = " + str(in_dict['adv']) +" / " + str(in_dict['decl'])))
    out_msg.append(("New High / New Low = " + str(in_dict['new_high']) +" / " + str(in_dict['new_low'])))
    out_msg.append(f"Gauge = {in_dict['gauge']:.2f}")
    out_msg.append(f"Stock above its 20-DMA (%):  {round(in_dict['c_20']/total*100,3):.2f}")
    out_msg.append(f"Stock above its 50-DMA (%):  {round(in_dict['c_50']/total*100,3):.2f}")
    out_msg.append(f"Stock with 20-DMA > 50-DMA (%):  {round(in_dict['s_20_50']/total*100,3):.2f}")
    out_msg.append(f"Stock with 50-DMA > 200-DMA (%):  {round(in_dict['s_50_200']/total*100,3):.2f}")
    out_msg.append(f"Stock with 50 > 150 > 200-DMA (%):  {round(in_dict['s_50_150_200']/total*100,3):.2f}")
    out_msg.append(f"Stock with rising 200-DMA (%):  {round(in_dict['s_200_200_20']/total*100,3):.2f}")
    out_msg.append("Number of Stock that fit the conditions: " + str(in_dict['stocks_fit_condition']))
    out_msg.append(f"Number of Stock that fit the conditions(%):  {round(in_dict['stocks_fit_condition']/total*100,3):.2f}")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Times", size=15)
    for msg in out_msg:
        pdf.cell(200, 10, txt = msg, ln = 1, align = 'L')
    pdf.cell(200, 10, txt = '', ln = 1)
    pdf.multi_cell(200,10, txt=str(instockname), border=0, align='L', fill=False)
    pdf.output(output_dir+"report_frontpage.pdf")

    # Also print the report in stdout line by line
    print("--------------------------------------------------------------------------------")
    print("------------------------ Stock Report for today --------------------------------")
    print("--------------------------------------------------------------------------------")
    for msg in out_msg:
        print(msg)


def gen_report_output_page(input_dir, output_dir):
    '''
    Convert the jpg charts in the input directory into a multi page PDF
    Write it to an output directory
    '''
    jpg_list = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]

    if len(jpg_list) == 0:
        print("No JPG found in input dir. Returning a blank PDF")
        pdf = FPDF()
        pdf.add_page()
        pdf.output(output_dir+'report_outputpage.pdf')
    else:
        with open(output_dir+'report_outputpage.pdf', 'wb') as f:
            f.write(img2pdf.convert([input_dir+filename for filename in sorted(jpg_list, key=lambda x: x[:6], reverse=True)]))
    return 1


def gen_report_breadth_page(in_dict, in_date, output_dir):
    '''
    Generate a breath plot pdf and write it to the output directory
    '''
    if in_dict['adv'] >= in_dict['decl']:
        use_color = 'green'
    else:
        use_color = 'red'

    bp_plot_arr = np.array(in_dict['breadth_per_list'])

    plt.style.use('seaborn')
    fig = plt.figure(figsize=(10,5))
    sns.histplot(bp_plot_arr[(bp_plot_arr < 20) & (bp_plot_arr > -20)],
                kde=False, bins=50, color=use_color, edgecolor=use_color, linewidth=0)

    # Text annotation
    nbreadth = (in_dict['adv'] - in_dict['decl']) / (in_dict['adv'] + in_dict['decl']) * 100
    anc = AnchoredText(f"{in_date}\nStocks Up:  {str(in_dict['adv'])}\nStocks Down:  {str(in_dict['decl'])}\nNet breadth: {nbreadth:.2f}%", \
                      loc="upper left", frameon=False, pad=0.5, prop=dict(size=12))
    ax = plt.gca()
    ax.add_artist(anc)
    ax.set_xlabel('% Change')
    ax.set_xlim(-20, 20)
    ax.set_ylabel('# of stock')

    fig.savefig(output_dir+'report_breadthpage.pdf', bbox_inches='tight')
    return 1


def gen_report_combine(input_dir, output_dir, daily_file_name):
    '''
    Combine the frontpage and outputpage into a single PDF
    '''
    f_file = open(input_dir+'report_frontpage.pdf', 'rb')
    b_file = open(input_dir+'report_breadthpage.pdf', 'rb')
    o_file = open(input_dir+'report_outputpage.pdf', 'rb')
    outpdf = output_dir+str(daily_file_name)+'.pdf'

    comb = PdfFileMerger()
    comb.append(PdfFileReader(f_file,strict=False))
    comb.append(PdfFileReader(b_file,strict=False))
    comb.append(PdfFileReader(o_file,strict=False))

    if os.path.isfile(outpdf):
        os.remove(outpdf)

    comb.write(outpdf)

    os.remove(input_dir+'report_frontpage.pdf')
    os.remove(input_dir+'report_breadthpage.pdf')
    os.remove(input_dir+'report_outputpage.pdf')

    f_file.close()
    o_file.close()
    return 1


def convert_report_dict_to_df(in_dict):
    '''
    Convert the input report_dict to a df with new column names

    report_dict ={'date':date_study, 'adv':0, 'decl':0, 'new_high':0, 'new_low':0,
                  'c_20':0, 'c_50':0, 's_20_50':0, 's_50_200':0,
                  's_200_200_20':0, 's_50_150_200':0, 'gauge':0, 'stocks_fit_condition':0,
                  'index_list':[], 'stock_ind_list':[]}
    '''
    total = in_dict['adv'] + in_dict['decl']
    data = {'Date': [in_dict['date']],
            'Number of stock': [total],
            'Advanced (Day)': [in_dict['adv']],
            'Declined (Day)': [in_dict['decl']],
            'New High': [in_dict['new_high']],
            'New Low': [in_dict['new_low']],
            'Gauge': [in_dict['gauge']],
            'Stock above 20-DMA': [round(in_dict['c_20']/total*100,3)],
            'Stock above 50-DMA': [round(in_dict['c_50']/total*100,3)],
            'Stock with 20-DMA > 50-DMA': [round(in_dict['s_20_50']/total*100,3)],
            'Stock with 50-DMA > 200-DMA': [round(in_dict['s_50_200']/total*100,3)],
            'Stock with 50 > 150 > 200-DMA': [round(in_dict['s_50_150_200']/total*100,3)],
            'Stock with 200-DMA is rising': [round(in_dict['s_200_200_20']/total*100,3)],
            'Number of Stock that fit condition': [in_dict['stocks_fit_condition']],
            'Number of Stock that fit condition(%)': [round(in_dict['stocks_fit_condition']/total*100,3)],
            'Tickers that fit the conditions': [in_dict['stock_ind_list']],
            'RS rating of Tickers': [in_dict['stock_rs_rating_list']],
            'RS rank of Tickers': [in_dict['stock_rs_rank_list']],
            'Breadth Percentage': [in_dict['breadth_per_list']]
            }
    df = pd.DataFrame(data)
    df['Date'] =pd.to_datetime(df.Date)
    return df



