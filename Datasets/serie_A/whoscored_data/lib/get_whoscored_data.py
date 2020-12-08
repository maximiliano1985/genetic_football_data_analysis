import codecs
import pandas as pd
import re
from time import sleep
from datetime import datetime as dt

from selenium import webdriver

def add_player(players, pl_data = ['', '', '', '', '', '']):
    players.loc[-1] = pl_data
    players.index = players.index + 1  # shifting index
    return players.sort_index()  # sorting by index

def parse_date(str_date):
    #print(str_date)
    day = re.findall(', ([0-9]*?)-', str_date)
    month = re.findall('-([a-zA-Z]*?)-', str_date)
    year = re.findall('-([0-9]*?)$', str_date)
    
    dict_months = {
            'Jan':'1',
            'Feb':'2',
            'Mar':'3',
            'Apr':'4',
            'May':'5',
            'Jun':'6',
            'Jul':'7',
            'Aug':'8',
            'Sep':'9',
            'Oct':'10',
            'Nov':'11',
            'Dec':'12'
        }                                 
    date = day[0]+'/'+dict_months[month[0]]+'/'+year[0]
    return dt.strptime(date, '%d/%m/%y').date()


def download_whoscored_scores(year_page, driver, end_month = 'May'):
    whoscored_addr = 'https://www.whoscored.com'

    driver.get(year_page);
    html = driver.page_source

    widget_calendar = re.findall('<dl class="listbox fixture-calendar">((.|\n)*?)</div>', html)[0][0]    
    month = re.findall('<span class="text">([a-zA-Z]*?)\W[0-9]*</span>', widget_calendar)[0]
    
    start_month = 'Aug'
    
    # rewind the pages
    while month != start_month:
        try:
            driver.find_element_by_css_selector('a.previous.button.ui-state-default.rc-l.is-default').click()
        except:
            start_month = 'Sep'
        sleep(2)
        html = driver.page_source
        widget_calendar = re.findall('<dl class="listbox fixture-calendar">((.|\n)*?)</div>', html)[0][0]
        month = re.findall('<span class="text">([a-zA-Z]*?)\W[0-9]*</span>', widget_calendar)[0]

    matches = None #pd.DataFrame(columns = ['date', 'HomeTeam', 'AwayTeam', 'team', 'coordinate', 'score'])
    init = True
    while True:
        # go forward
        print(month+'_'+year_page[-9:])
    
        # loop through all the matches of the month
        html = driver.page_source
        table = re.findall('<table id="tournament-fixture" class="grid hover fixture">(.*?)</tbody>', html)[0]
        match_links = re.findall('<td class="toolbar right"><a href="(.*?)" class="match-link match-report rc">Match Report</a>', table)
    
        for link in match_links:
            url = whoscored_addr + link.replace('MatchReport', 'Live')
            print(url)
            driver.get(url);
            sleep(2)
            page = driver.page_source
            sleep(1)
            match = parse_page(page)
            #print(match)   
            
            if init:
                matches = match
                init = False
            else:
                matches = pd.concat([matches, match], ignore_index=True)

    
        if month == end_month:
            break
        else:
            # back to the main page
            driver.get(year_page);
            # rewind the pages
            completed_month = month
            iter_month = ''
            while iter_month != completed_month:
                driver.find_element_by_css_selector('a.previous.button.ui-state-default.rc-l.is-default').click()
                    
                sleep(2)
                html = driver.page_source
                widget_calendar = re.findall('<dl class="listbox fixture-calendar">((.|\n)*?)</div>', html)[0][0]
                iter_month = re.findall('<span class="text">([a-zA-Z]*?)\W[0-9]*</span>', widget_calendar)[0]    
    
            driver.find_element_by_css_selector('a.next.button.ui-state-default.rc-r.is-default').click()
            sleep(2)
            html = driver.page_source
            widget_calendar = re.findall('<dl class="listbox fixture-calendar">((.|\n)*?)</div>', html)[0][0]    
            month = re.findall('<span class="text">([a-zA-Z]*?)\W[0-9]*</span>', widget_calendar)[0]

    return matches
    



def download_web_page(url, annotation=''):
    driver = webdriver.Chrome('./lib/chromedriver')  # Optional argument, if not specified will search path.
    driver.get(url);
    html = driver.page_source

    filename = './pages/'+url.split('/')[-1]+annotation+'.htm'
    url.split('/')
    f = open( filename, "wt" )
    f.write(html)
    f.close()

    driver.close()

    return filename

def parse_page(page):
    players = pd.DataFrame(columns = ['date', 'HomeTeam', 'AwayTeam', 'team', 'coordinate', 'score'])
    HAteamsTok = []
        
    tok = re.findall('Date:</dt><dd>(.*?)</dd></dl>', page)
    if tok != []:
        dateTok = parse_date(tok[0])
        
    if HAteamsTok == []:
        HAteamsTok = re.findall('class="team-name">(.*?)</a>', page)
            
    pitchTok = re.findall('<div class="pitch"(.*?)</div></div></div></div></div>', page)
        
    for pthTok in pitchTok:
        playerTok = re.findall('<div class="player(.*?)<div class="player-info">', pthTok)

        for plTok in playerTok:
            #print(plTok)
            teamTok = re.findall('data-field="(.*?)"', plTok)
            
            if teamTok[0] == 'home':
                coordTok = re.findall('left: (.*?)%', plTok)
            else:
                coordTok = re.findall('right: (.*?)%', plTok)
    
            scoreTok = re.findall('px;">(.*?)</span></div>', plTok)
            players = add_player(players, [dateTok, HAteamsTok[0], HAteamsTok[1], teamTok[0], coordTok[0], scoreTok[0]])

    # Check the proper parsing of the html file
    #assert (len(players) == 22), 'STAT_EXTRACTION_ERROR: found only {} players'.format(len(players))
    if (len(players) != 22):
        print('STAT_EXTRACTION_WARNING: found only %i players' %(len(players)) )
    return players


def parse_page_addr(addr):
    page = codecs.open(addr, 'r')

    players = pd.DataFrame(columns = ['date', 'HomeTeam', 'AwayTeam', 'team', 'coordinate', 'score'])
    HAteamsTok = []
    for line in page:
        tok = re.findall('Date:</dt><dd>(.*?)</dd></dl>', line)
        if tok != []:
            dateTok = parse_date(tok[0])
        
        if HAteamsTok == []:
            HAteamsTok = re.findall('class="team-name">(.*?)</a>', line)
            
        pitchTok = re.findall('<div class="pitch"(.*?)</div></div></div></div></div>', line)
        
        for pthTok in pitchTok:
            playerTok = re.findall('<div class="player(.*?)<div class="player-info">', pthTok)

            for plTok in playerTok:
                #print(plTok)
                teamTok = re.findall('data-field="(.*?)"', plTok)
            
                if teamTok[0] == 'home':
                    coordTok = re.findall('left: (.*?)%', plTok)
                else:
                    coordTok = re.findall('right: (.*?)%', plTok)
    
                scoreTok = re.findall('px;">(.*?)</span></div>', plTok)
                players = add_player(players, [dateTok, HAteamsTok[0], HAteamsTok[1], teamTok[0], coordTok[0], scoreTok[0]])

    # Check the proper parsing of the html file
    #assert (len(players) == 22), 'STAT_EXTRACTION_ERROR: file <{}>, found only {} players'.format(addr, len(players))
    if (len(players) != 22):
        print('STAT_EXTRACTION_WARNING: found only %i players' %(len(players)) )
    return players