#!/usr/bin/env python
# coding: utf-8

# Format knižnica na prácu so súborom pre Kopija_Finder_6.0

import os as os
import numpy as np

directory = None
file = None

def info():
    print("Format knižnica na prácu so súborom pre Kopija_Finder_6.0")

    
# Vytvori folder output ak neexistuje

def make_folder(directory=directory):
    try:
        os.mkdir(directory)
    except OSError:
        print("Vytvorenie priečinka neuskutocnene %s pravdepodobne existuje" % directory)
    else:
        print("Úspešne vytvorenie priečinka %s " % directory)
        
        
# Načíta časový údaj začiatku merania

def start_data(file_start):
    start = open(file_start, "r")
    print("Čas začiatku merania:")
    print("\t" + start.readline())
    

# Načítanie súborov "time" a "mag" do listu a vypíše počet údajov v nich,
# následne ich zmení do np.array

def time_data(file_time):
    time = np.loadtxt(file_time, dtype=np.float64)
    print("Počet časových údajov\t\t=\t" + str(len(time)))
    return time

#     with open(file_time, "r") as time: 
#         time = time.readlines()
#         print("Počet časových údajov\t\t=\t" + str(len(time)))
#         return np.array(time, dtype=np.float64)


def mag_data(file_mag):
    mag = np.loadtxt(file_mag, dtype=np.float64)
    print("Počet magnetických údajov\t=\t" + str(len(mag)))
    return mag

#     with open(file_mag, "r") as mag:
#         mag = mag.readlines()
#         print("Počet magnetických údajov\t=\t" + str(len(mag)))
#         return np.array(mag, dtype=np.float64)

    
# Uloženie dát

def save_data(peaks_time, peaks_mag, directory=directory, file=file):
    last_t = None
    with open(directory + "/output/output_" + file + ".txt", "w") as output_doc:
        for t, m in zip(peaks_time, peaks_mag):
            if last_t == t:
                continue
            else:
                output_doc.write(str(t) + "\t" + str(m) + "\n")
                last_t = t
                

def save_txt(peaks_time, peaks_mag, directory=directory, file=file):
    np.savetxt(directory + "/output/output_" + file + ".txt", np.transpose(np.array([np.unique(peaks_time), np.unique(peaks_mag)])), delimiter='\t', newline="\n")


def save_csv(peaks_time, peaks_mag, directory=directory, file=file):
    np.savetxt(directory + "/output/output_" + file + ".csv", np.transpose(np.array([np.unique(peaks_time), np.unique(peaks_mag)])), delimiter=',', newline="\n")
               
                
# Funkcie pre čítanie jedného súboru

def load(file=file, directory=directory):
    file_mag = directory + file + "_mag.txt"
    file_time = directory + file + "_time.txt"
    file_start = directory + file + "_start.txt"

    # Vypíše časový údaj o začiatku daného merania
    start_data(file_start)

    # Vypise pocty údajov
    time = time_data(file_time)
    mag = mag_data(file_mag)
    n = len(mag)
    
    return time, mag, n

# Funkcie pre čítanie viecerých súborov

# Ukáže charakteristiku priečinka s dátami


def intro(directory=directory):
    # Vypíše len všetko čo je v priečinku
    print(directory)
    list_files = os.listdir(directory)
    print("\nCelkový počet súborov v priečinku:", str(len(list_files)))
    for file in list_files:
        print("\t" + file)

    # Vyberie iba súbory s príponou .txt
    txt_files = set()
    for file in list_files:
        if ".txt" in file:
            txt_files.add(file)

    # Extrahuje aké kódy meraní sa nachádzajú v priečinku
    merania = []
    for file in txt_files:
        file = file.split("_")
        merania.append(file[0])

    # Vypíše aké marania sa v priečinku nachádzajú a ich počet
    merania = set(merania)
    print("\nCelkový počet meraní v priečinku:", str(len(merania)))
    for meranie in merania:
        print("\t" + meranie)

    print(60*"_")
    return list(merania)