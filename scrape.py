from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import time
import pickle
import numpy as np
import progressbar
import logging

# configure logging
logging.basicConfig(filename='scrape.log', filemode='a+',
                    format='%(levelname)s - %(message)s')

def scrape_link_info(row, case, verbose=False):
    """
    Scrapes artist/song/link-to-audio info for a given case. Given the row object
    for a given case, this finds the link to the specific case page, requests
    the html and obtains the artist(s)/song(s)/link(s)-to-audio data.

    Parameters:
        row (BeautifulSoup object): soup object for the row of the case in
            question.
        case (str): case name. This name will appear as the first entry of each
            row and is the basis upon which to join the two dataframes.
        verbose (bool): theoretically adds print statements to check what's going
            on, but in actuality this currently does nothing.

    Returns:
        df (pandas dataframe): constructs a pandas dataframe of the scraped
            content. Columns are formatted as:
             __________________________________________________________
            | case-name | complaining | artist | title | link-to-audio |
            |-----------+-------------+--------+-------+---------------|
            |    str    |     bool    |  str   |  str  |      str      |
            |___________|_____________|________|_______|_______________|

            There is no index column.
    """
    # get link to page with audio
    link = row.find(class_='column-3')
    link = link.find(href=re.compile(r'http')).get('href')

    # request page that has audio links on it
    audio_page = requests.get(link).text
    soup = BeautifulSoup(audio_page, 'html.parser')

    try:
        # get complaining artist/defending artist columns
        comp = soup.find_all(class_='ms-rteTableEvenCol-default')[1] # only second
        defn = soup.find_all(class_='ms-rteTableOddCol-default')[1] # entry needed
    except IndexError:
        # if no second entry, case doesn't have requisite information
        return None

    # get data from each complaining/defending artist columns
    comp_data = comp.find_all(class_='ms-rteElement-P')
    defn_data = defn.find_all(class_='ms-rteElement-P')

    # lists to hold cleaned complaining/defending artist/title/link info
    clean_data = []
    clean_temp = []
    temp_link = ''

    entry_num = 0       # tracks artist/title/link groups
    comp_bool = False   # tracks if we are in the comp_data or defn_data set

    for d_group in [comp_data, defn_data]:
        # true when in comp_data iter, false when in defn_data iter
        comp_bool = not comp_bool
        clean_temp = [] # this should already be empty, but *just* in case
        group_len = len(d_group) # to track when we should append link and break
        end_group = False        # binary switch to indicate end of group

        for item in d_group:
            if len(clean_temp) == 0:
                # add case title
                clean_temp.append(case)
                # add bool to track if this is complaining/defending data
                clean_temp.append(comp_bool)

            try:
                # if this contains the hyperlink, get it
                href_temp = item.find(href=re.compile(r'http'))

                if re.match('([Hh]ear|[Aa]udio)', href_temp.text):
                    # if this link contains some sort of audio data, get link
                    cur_link = href_temp.get('href')
                    clean_temp.append(cur_link)
                    clean_data.append(clean_temp)
                    clean_temp = []
                    entry_num = -1

                    # if temp_link:
                    #     # if we've already collect a link, append with comma sep
                    #     temp_link += ',' + cur_link
                    # else:
                    #     temp_link += cur_link

            except AttributeError:
                # if this isn't something with a hyperlink, collect its data
                if entry_num == 0 or entry_num == 1:
                    text_temp = item.text

                # check if this is the '—' entry between artist/title/link groups
                    # note that that is a *long* dash
                if text_temp == '—' or text_temp == '\n':
                    # if we are here, then this is the end of the current sub-group
                    if len(clean_temp) == 4:
                        clean_temp.append(None)
                        clean_data.append(clean_temp)
                        clean_temp = []
                        entry_num = -1
                else:
                    # if this is artist/title data, add it
                    clean_temp.append(text_temp)
            finally:
                entry_num += 1

            # if end_group:
            #     clean_temp.append(temp_link)
            #     clean_data.append(clean_temp)
            #     clean_temp = []
            #     temp_link = ''
            #     entry_num = -1


    # now convert/return the data in clean_data into a pandas dataframe
    columns = ['case', 'complaining', 'artist', 'title', 'link']
    return pd.DataFrame(clean_data, columns=columns)

def base_scrape(base_url='https://blogs.law.gwu.edu/mcir/cases/', link_df=None,
                gen_df=None, save=True, wait_time=1, verbose=False, v_verbose=False):
    """
    """
    gen_columns = ['year', 'country', 'case', 'complaining_work',
                   'defending_work', 'complaining_author', 'defending_author']
    link_columns = ['case', 'complaining', 'artist', 'title', 'link']

    if link_df is None:
        # if we didn't pass in a link_df to add to, initialize empty one
        link_df = pd.DataFrame(columns=link_columns)
    if gen_df is None:
        # if we didn't pass in a gen_df to update, initialize an empty one
        gen_df = pd.DataFrame(columns=gen_columns)

    # request source
    source = requests.get(base_url).text
    # soup object
    soup = BeautifulSoup(source, 'html.parser')
    # get the table of stuff
    row_hover = soup.find(class_='row-hover')
    # grab each row
    rows = row_hover.find_all(class_=re.compile(r'row-[\d]+ (even|odd)'))

    if verbose:
        # if we want to know where we're at, use a progressbar
        rows = progressbar.progressbar(rows)

    # go through each row and get the information
    for row in rows:
        # pause for a moment before moving on
        time.sleep(wait_time)
        # get the columns that contain what we want
        cols = row.find_all(class_=re.compile(r'column-[\d]+'))

        # try to extract information, if anything breaks log what broke and skip
        try:
            # extract all the information, filter to what we want
            info = [item.text for item in cols]
            info = info[:3] + info[4:]

            # pause for a moment, then update link dataframe
            link_df_temp = scrape_link_info(row, info[2], verbose=v_verbose)

            if link_df_temp is None:
                continue

        except KeyboardInterrupt as e:
            raise e

        except Exception as e:
            logging.exception("Exception occured")
            continue

        # if everything went well, update our dataframes
        link_df = link_df.append(link_df_temp)
        gen_df = gen_df.append(pd.DataFrame([info], columns=gen_columns))

        if v_verbose:
            print(link_df)
            print('\n')
            print(gen_df)
            print('\n\n')

    # save the data
    if save:
        with open('data/gen_df', 'wb') as f:
            pickle.dump(gen_df, f)
        with open('data/link_df', 'wb') as f:
            pickle.dump(link_df, f)

    return gen_df, link_df


def scrape_audio():
    pass
