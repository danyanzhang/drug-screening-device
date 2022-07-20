import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
import statistics as stats
import synergy
import os

class Device():
    '''Takes input concentrations, aligns to concentration grid'''
    
    #drug1_input
    #drug2_input
    #drgu3_input

    def __init__(self, dataTable, drug1_max, drug2_max, drug3_max, units):
        '''Create a drug Device object with a given dataTable of cell information,
        and maximum starting drug concentrations.'''
        self.drug1_input = drug1_max
        self.drug2_input = drug2_max
        self.drug3_input = drug3_max
        self.dataTable = dataTable # pandas dataframe
        self.units = units # string of concentration units (i.e. nM, uM)

    @classmethod
    def from_excel(cls, filename, drug1_max, drug2_max, drug3_max, units):
        df = pd.read_excel(filename, header=None)
        df.rename({0: "x", 1: "y", 2: "c1", 3: "c2", 4: "c3", 5: "live"}, axis=1, inplace=True)
        return cls(df, drug1_max, drug2_max, drug3_max, units)

    def print_summary(self):
        print(f"this is a test of {self.drug1_input}")
        print(f"Concentration Units: {self.units}")

    def zone_viability(filtered_data_frame):
        '''get viability of a specified zone'''

    def get_zone(self, zone):
        df1 = self.dataTable
        df_filt = df1.copy()
        zone_threshold = 0.68
        combo_threshold = 0.45
        drug3_threshold = 0.27
        plotting = False

        match zone:
            case 1:
                df_filt = df_filt[(df_filt["c1"] > zone_threshold) & (df_filt["c2"] > ((1-zone_threshold)/2))]
                #print(f"Drug 1 max on drug 2 side. {len(df_filt)} cells counted.")
            case 2:
                df_filt = df_filt[(df_filt["c1"] > zone_threshold) & (df_filt["c3"] > ((1-zone_threshold)/2))]
                #print(f"Drug 1 max on drug 3 side. {len(df_filt)} cells counted.")
            case 3:
                df_filt = df_filt[(df_filt["c2"] > zone_threshold) & (df_filt["c1"] > ((1-zone_threshold)/2))]
                #print(f"Drug 2 max on drug 1 side. {len(df_filt)} cells counted.")
            case 4:
                df_filt = df_filt[(df_filt["c2"] > zone_threshold) & (df_filt["c3"] > ((1-zone_threshold)/2))]
                #print(f"Drug 2 max on drug 3 side. {len(df_filt)} cells counted.")
            case 5:
                df_filt = df_filt[(df_filt["c3"] > zone_threshold) & (df_filt["c1"] > ((1-zone_threshold)/2))]
                #print(f"Drug 3 max on drug 1 side. {len(df_filt)} cells counted.")
            case 6:
                df_filt = df_filt[(df_filt["c3"] > zone_threshold) & (df_filt["c2"] > ((1-zone_threshold)/2))]
                #print(f"Drug 3 max on drug 2 side. {len(df_filt)} cells counted.")
            case 7: # drug 12 middle
                df_filt = df_filt[(df_filt["c1"] > combo_threshold) & (df_filt["c2"] > combo_threshold)]
                #print(f"Drug 1 and 2 combination. {len(df_filt)} cells counted.")
            case 8:
                df_filt = df_filt[(df_filt["c1"] > combo_threshold) & (df_filt["c3"] > combo_threshold)]
                #print(f"Drug 1 and 2 combination. {len(df_filt)} cells counted.")
            case 9:
                df_filt = df_filt[(df_filt["c2"] > combo_threshold) & (df_filt["c3"] > combo_threshold)]
                #print(f"Drug 1 and 2 combination. {len(df_filt)} cells counted.")
            case 10:
                df_filt = df_filt[(df_filt["c1"] > drug3_threshold) & (df_filt["c2"] > drug3_threshold) & (df_filt["c3"] > drug3_threshold)]
                #print(f"Drug 1 and 2 combination. {len(df_filt)} cells counted.")
            case 11:
                df_filt = df_filt[(df_filt["c1"] > 0.6) & (df_filt["c2"] > (1-0.6)-0.07) & (df_filt["c3"] < 0.05)]
                #print(f"Drug 1 and 2 combination. {len(df_filt)} cells counted.")
            case 12:
                df_filt = df_filt[(df_filt["c2"] > 0.6) & (df_filt["c1"] > (1-0.6)-0.07) & (df_filt["c3"] < 0.05)]
                #print(f"Drug 1 and 2 combination. {len(df_filt)} cells counted.")
                

        #df_filt = df_filt[(df_filt["c1"] > zone_threshold)]
        #print(f"Total number of cells: {len(df1)}")
        #print(f"Filtered condition: {len(df_filt)}")
        viability_max = df_filt["live"].mean()
        c1_avg = df_filt["c1"].mean()
        c2_avg = df_filt["c2"].mean()
        c3_avg = df_filt["c3"].mean()
        #print(f"Viability of section: {viability_max}")

        #print(f"Viability of section: {viability_max_1}")

        # shows entire space
        if plotting == True:
            print(df_filt["c1"].mean())
            print(df_filt["c2"].mean())
            print(df_filt["c3"].mean())

            
            fig, ax = plt.subplots(2)
            ax[0].set_aspect('equal')
            ax[0].scatter(df1["x"], df1["y"])
            ax[0].scatter(df_filt["x"], df_filt["y"])
            ax[1].set_aspect('equal')
            #ax[1].scatter()
            # test ternary plot
            ternx, terny = terncoords(df1[["c1", "c2", "c3"]])
            ax[1].scatter(ternx, terny)
            ternx_filt, terny_filt = terncoords(df_filt[["c1", "c2", "c3"]])
            ax[1].scatter(ternx_filt, terny_filt)
            plt.show()
        return viability_max, c1_avg, c2_avg, c3_avg

    def zone_ratio(ratio):
        '''define a zone for a specific ratio of drugs.
        this will be used to correct non-equipotent ratios'''
        #return filtered_data_frame

    def plot_zone():
        '''plot stuff'''

    def plot_ternary():
        '''plot a ternary plot'''

    def plot_smile():
        '''plot smile plot visualization'''
    
    def viability_zone():
        '''get viability from a specific zone'''

    def plot_region(self, df):
        # shows entire space
        fig, ax = plt.subplots(2)
        ax[0].set_aspect('equal')
        ax[0].scatter(self.dataTable["x"], self.dataTable["y"])
        ax[0].scatter(df["x"], df["y"])
        ax[1].set_aspect('equal')
        #ax[1].scatter()
        # test ternary plot
        ternx, terny = terncoords(self.dataTable[["c1", "c2", "c3"]])
        ax[1].scatter(ternx, terny)
        ternx_filt, terny_filt = terncoords(df[["c1", "c2", "c3"]])
        ax[1].scatter(ternx_filt, terny_filt)
        plt.show()

    def get_zone_ratio(self, target_ratio, drug1_idx, drug2_idx, plotting = False):
        '''ratio is the x:1 ratio of drug c1:c2
        Uses a linear search algorithm to find the zone with target ratio
        Drug 1 and drug 2 are just the indexes of which drug 1, 2, or 3
        '''
        print(f"Ratio: {target_ratio}")
        df = self.dataTable.copy()
        zone_threshold = 0.68
        combo_threshold = 0.45
        drug3_threshold = 0.27
        window_size = 0.08

        max_attempts = 100
        attempts = 0

        c1_start = 0.3
        error = 100 # large value guaranteed to trigger
        #target_ratio = 1.4 # starting
        error_thresh = 0.01 # if within 1%

        while (abs(error) > error_thresh) & (attempts <= max_attempts):
            
            if (drug1_idx == 1) & (drug2_idx == 2):
                df_filt = df[(df["c1"] > c1_start) & (df["c2"] > (1-c1_start)-window_size) & (df["c3"] < 0.10)]
            elif (drug1_idx == 1) & (drug2_idx == 3):
                df_filt = df[(df["c1"] > c1_start) & (df["c3"] > (1-c1_start)-window_size) & (df["c2"] < 0.10)]
            elif (drug1_idx == 2) & (drug2_idx == 3):
                df_filt = df[(df["c2"] > c1_start) & (df["c3"] > (1-c1_start)-window_size) & (df["c1"] < 0.10)]
            else:
                print("Some error in assigning drug indexes")
            
            #print(f"{len(df_filt)} cells counted in region")
            viability_max = df["live"].mean()
            c1_avg = df_filt["c1"].mean()
            c2_avg = df_filt["c2"].mean()
            c3_avg = df_filt["c3"].mean()
            c_avg = [c1_avg, c2_avg, c3_avg] # make a list
            #print(f"C1 avg: {c1_avg:.2f}")
            #print(f"C2 avg: {c2_avg:.2f}")
            #print(f"C3 avg: {c3_avg:.2f}")
            ratio_12 = c_avg[drug1_idx-1] / c_avg[drug2_idx-1]

            #print(f"Viability of section: {viability_max:.2f}")
            #print(f"Drug ratio C1/C2: {ratio_12:.2f}")

            error = (ratio_12 - target_ratio)/target_ratio
            #print(f"Error: {error:.2f}")

            c1_start -= np.sign(error) * 0.01
            #print(f"\nNew starting value = {c1_start:.2f}")

            attempts += 1 # iterate attempts
            if plotting == True: self.plot_region(df_filt)
        else:
            print("Success!")

        print(f"Viability of section: {viability_max:.2f}")
        print(f"Drug ratio C1/C2: {ratio_12:.2f}")
        return df_filt["live"].mean(), c_avg[drug1_idx-1], c_avg[drug2_idx-1]

class DeviceStack():
    '''A collection of devices that'''
    #num_devices
    #drug1_concentrations
    #drug2_concentrations
    #drug3_concentrations
    #drug1_name
    #drug2_name
    #drug3_name
    def __init__(self, folder_name):
        working_dir = os.getcwd()

        folder_path = os.path.join(working_dir, folder_name) # assuming it is a subfolder of working directory
        # should change this assumption at some point if I'm packaging it as a standalone

        #print(lines) # lines object is just a list
        file_list = []
        for file in sorted(os.listdir(folder_path)):
            if file.endswith(".xlsx"):
                file_list.append(file)
                #print(os.path.join(folder_path, file))

        # drug info text file
        with open(os.path.join(folder_path, 'info.txt')) as f:
            lines = f.readlines()

        # drug 1 info
        info_drug1_line = lines[0]
        info_drug1_split = info_drug1_line.split(';')
        self.drug1_name = info_drug1_split[0].strip()
        self.drug1_doses = np.array(eval(info_drug1_split[1].strip()))
        self.drug1_units = info_drug1_split[2].strip()

        # drug 2 info
        info_drug2_line = lines[1]
        info_drug2_split = info_drug2_line.split(';')
        self.drug2_name = info_drug2_split[0].strip()
        self.drug2_doses = np.array(eval(info_drug2_split[1].strip()))
        self.drug2_units = info_drug2_split[2].strip()

        # drug 3 info
        info_drug3_line = lines[2]
        info_drug3_split = info_drug3_line.split(';')
        self.drug3_name = info_drug3_split[0].strip()
        self.drug3_doses = np.array(eval(info_drug3_split[1].strip()))
        self.drug3_units = info_drug3_split[2].strip()

        self.devices = []
        i = 0
        for file in file_list:
            self.devices.append(Device.from_excel(os.path.join(folder_path, file),\
                self.drug1_doses[i], self.drug2_doses[i], self.drug3_doses[i], self.drug1_units))
            i += 1

    def __str__(self):
        return "DeviceStack:{}-{}-{}".format\
            (self.drug1_name[0:4], self.drug2_name[0:4], self.drug3_name[0:4])


    def get_num_devices(self):
        '''get the number of devices in stack'''
        return len(self.devices)

    def diamond(drug1, drug2):
        '''get diamond synergy between any 2 drugs'''

    def bliss(drug1, drug2):
        '''get bliss synergy between any 2 drugs'''

    def single_drug_fit(self, drug_idx, plotting = False):
        '''Get single drug fit of drug, at specific condition
        Side 1, Side 2, or Average of the two'''
        
        viability1 = np.empty(self.get_num_devices())
        viability2 = np.empty(self.get_num_devices())
        dose_actual_1 = np.empty(self.get_num_devices()) # because the actual doses may not be the same
        dose_actual_2 = np.empty(self.get_num_devices())

        match drug_idx:
            case 1:
                zone1, zone2 = 1, 2
                doses = self.drug1_doses
                drug_name = self.drug1_name
                output_idx = 1
            case 2:
                zone1, zone2 = 3, 4
                doses = self.drug2_doses
                drug_name = self.drug2_name
                output_idx = 2
            case 3:
                zone1, zone2 = 5, 6
                doses = self.drug3_doses
                drug_name = self.drug3_name
                output_idx = 3
        

        for i in range(self.get_num_devices()):
            viability1[i] = self.devices[i].get_zone(zone1)[0] # only get first value
            #print(self.devices[i].get_zone(1))
            viability2[i] = self.devices[i].get_zone(zone2)[0]
            dose_actual_1[i] = self.devices[i].get_zone(zone1)[output_idx]
            dose_actual_2[i] = self.devices[i].get_zone(zone2)[output_idx]

        # multiply doses by their starting input
        dose_actual_1 = np.multiply(dose_actual_1, doses)
        dose_actual_2 = np.multiply(dose_actual_2, doses)


        if plotting == True:
            # plotting
            plt.scatter(doses, dose_actual_1)
            plt.scatter(doses, dose_actual_2)
            plt.ylabel("Actual Dose")
            plt.xlabel("Input Dose")
            plt.show()

        
        # hill function for position 1
        hill_1 = synergy.single.Hill()
        #print(dose_actual_1)
        #print(viability1)
        hill_1.fit(dose_actual_1, viability1)

        # hill function for position 2
        hill_2 = synergy.single.Hill()
        hill_2.fit(dose_actual_2, viability2)
        
        # aggregate of both
        hill_agg = synergy.single.Hill()
        doses_all = np.concatenate((dose_actual_1, dose_actual_2))
        viability_all = np.concatenate((viability1, viability2))

        hill_agg.fit(doses_all, viability_all)

        # hill function for aggregate
        EC50_1 = hill_1.get_parameters()[3]
        EC50_2 = hill_2.get_parameters()[3]
        EC50_agg = hill_agg.get_parameters()[3]

        if plotting == True:
            # plotting
            dose_fit = np.linspace(doses_all.min(), doses_all.max())
            E1 = hill_1.E(dose_fit)
            E2 = hill_2.E(dose_fit)
            E3 = hill_agg.E(dose_fit)
            fig, ax = plt.subplots()
            
            ax.scatter(dose_actual_1, viability1, label = 'Side 1 points')
            ax.plot(dose_fit, E1, label = 'Side 1 fit')
            ax.scatter(dose_actual_2, viability2, label = 'Side 2 points')
            ax.plot(dose_fit, E2, label = 'Side 2 fit')
            ax.plot(dose_fit, E3, label = 'Aggregate fit')
            #ax.set_xscale('log')
            ax.set_xlabel('Concentration [nM]')
            ax.set_ylabel('Viability')
            ax.set_ylim([0, 1])
            ax.legend()
            plt.title(f"Single Drug Fit for {drug_name}")
            plt.show()

        print(f"EC50_1: {EC50_1:.2f}, EC50_2: {EC50_2:.2f}, EC50_both: {EC50_agg:.2f}")
        #print(f"Median EC50: {stats.median([EC50_1, EC50_2, EC50_agg]):.2f}")
        
        return EC50_1, EC50_2, EC50_agg

    def combo_test(self, drug1_idx, drug2_idx):
        
        # get EC50 of drug 1 & 2
        EC50_1, EC50_2, EC50_agg = self.single_drug_fit(drug1_idx)
        drug1_EC50 = stats.median([EC50_1, EC50_2, EC50_agg])
        EC50_1, EC50_2, EC50_agg = self.single_drug_fit(drug2_idx)
        drug2_EC50 = stats.median([EC50_1, EC50_2, EC50_agg])

        match drug1_idx:
            case 1:
                doses1 = self.drug1_doses
                drug1_name = self.drug1_name
            case 2:
                doses1 = self.drug2_doses
                drug1_name = self.drug2_name
            case 3:
                doses1 = self.drug3_doses
                drug1_name = self.drug3_name
        
        match drug2_idx:
            case 1:
                doses2 = self.drug1_doses
                drug2_name = self.drug1_name
            case 2:
                doses2 = self.drug2_doses
                drug2_name = self.drug2_name
            case 3:
                doses2 = self.drug3_doses
                drug2_name = self.drug3_name
        

        target1, target2 = drug1_EC50/2, drug2_EC50/2
        print(f"Target EC50 for drug 1: {target1:.2f}")
        print(f"Target EC50 for drug 2: {target2:.2f}")

        # I now have to find a device in which the targets are matched up and can use a ratio
        # Translate the actual device concentrations into multiples of EC50?
        print(f"Normalized EC50 drug1: {np.around(doses1/drug1_EC50, 2)}")
        print(f"Normalized EC50 drug2: {np.around(doses2/drug2_EC50, 2)}")

        dose1_norm = doses1 / drug1_EC50
        dose2_norm = doses2 / drug2_EC50

        ratio = 1/np.mean(dose1_norm / dose2_norm)
        print(f"Ratio is: {ratio:.2f}")

        EC50_combo = self.ratio_zone_fit(ratio, drug1_idx, drug2_idx, dose1_norm, dose2_norm)
        print(f"Combination of {drug1_name} + {drug2_name}")
        print(f"EC50 of combo is: {EC50_combo:.2f}")

        return EC50_combo


    def _get_normalized_EC50(drug_idx):
        pass
    
    def ratio_zone_fit(self, ratio, drug1_idx, drug2_idx, dose1_norm, dose2_norm):
        '''Fits the zone indicated by the specific ratio.
        Also needs the normalized EC50 doses of drugs 1 and 2.
        Change this to calculate the normalized EC50 doses of drug 1 and 2.'''
        

        viability = np.empty(self.get_num_devices())
        c1 = np.empty(self.get_num_devices())
        c2 = np.empty(self.get_num_devices())
        plotting = False

        for i in range(self.get_num_devices()):
            viability[i], c1[i], c2[i] = self.devices[i].get_zone_ratio(ratio, drug1_idx, drug2_idx)

        rescaled_EC50 = ((np.multiply(c1, dose1_norm) + np.multiply(c2, dose2_norm))/2)*2
        # do I need to divide this by 2 again? I think so: or is it multiply? Let's try multiply
        print(np.around(rescaled_EC50,2))

        hill_combo = synergy.single.Hill()
        hill_combo.fit(rescaled_EC50, viability)
        EC50_combo = hill_combo.get_parameters()[3]

        return EC50_combo

def terncoords(df):
    '''Takes a pandas dataframe input of 3 columns and turns into ternary coordinates.'''
    a = df["c1"].to_numpy()
    b = df["c2"].to_numpy()
    c = df["c3"].to_numpy()
    ternx = (1/2) * (2*b + c) / (a + b + c)
    terny = (math.sqrt(3)/2) * c / (a + b + c)
    return ternx, terny


def combo():
    # got to be a better way to automate all of this
    df1 = pd.read_excel('Results_06032022_Dev1.xlsx', header=None)
    df2 = pd.read_excel('Results_06032022_Dev2.xlsx', header=None)
    df3 = pd.read_excel('Results_06032022_Dev3.xlsx', header=None)
    df4 = pd.read_excel('Results_06032022_Dev4.xlsx', header=None)
    df5 = pd.read_excel('Results_06032022_Dev5.xlsx', header=None)
    df6 = pd.read_excel('Results_06032022_Dev6.xlsx', header=None)
    #df8 = pd.read_excel('Results_06032022_Dev8.xlsx', header=None)

    data = [df1, df2, df3, df4, df5, df6]
    for df in data:
        df.rename({0: "x", 1: "y", 2: "c1", 3: "c2", 4: "c3", 5: "live"}, axis=1, inplace=True)

    viability12_zone10 = np.empty(len(data))
    viability12_zone11 = np.empty(len(data))
    viability12_zone12 = np.empty(len(data))

    for i in range(len(data)):
        viability12_zone10[i] = get_zone(data[i],10)
        viability12_zone11[i] = get_zone(data[i],11)
        viability12_zone12[i] = get_zone(data[i],12)


    dose_daun = np.array([400, 200, 100, 50, 25, 12.5])
    dose_nel = np.array([400, 200, 100, 50, 25, 12.5])
    dose_vcr = np.array([40, 20, 10, 5, 2.5, 1.25])

    # fit 
    hill_10 = synergy.single.Hill()
    hill_11 = synergy.single.Hill()
    hill_12 = synergy.single.Hill()
    hill_10.fit(dose_daun, viability12_zone10)
    hill_11.fit(dose_daun, viability12_zone11)
    hill_12.fit(dose_daun, viability12_zone12)

    print(hill_10.get_parameters())
    print(hill_11.get_parameters())
    print(hill_12.get_parameters())


    dose_daun_fit = np.linspace(dose_daun.min(), dose_daun.max())
    E_10 = hill_10.E(dose_daun_fit)
    E_11 = hill_11.E(dose_daun_fit)
    E_12 = hill_12.E(dose_daun_fit)

    # plotting
    fig, ax = plt.subplots()
    ax.scatter(dose_daun, viability12_zone10)
    ax.plot(dose_daun_fit, E_10)
    ax.scatter(dose_daun, viability12_zone11)
    ax.plot(dose_daun_fit, E_11)
    ax.scatter(dose_daun, viability12_zone12)
    ax.plot(dose_daun_fit, E_12)
    #ax.set_xscale('log')
    ax.set_xlabel('Concentration [nM]')
    ax.set_ylabel('Viability')
    ax.set_ylim([0, 1])
    plt.show()


def rep2():
    # got to be a better way to automate all of this
    df1 = pd.read_excel('Results_06062022_Dev1.xlsx', header=None)
    df2 = pd.read_excel('Results_06062022_Dev2.xlsx', header=None)
    df3 = pd.read_excel('Results_06062022_Dev3.xlsx', header=None)
    df4 = pd.read_excel('Results_06062022_Dev4.xlsx', header=None)
    df5 = pd.read_excel('Results_06062022_Dev5.xlsx', header=None)
    df6 = pd.read_excel('Results_06062022_Dev6.xlsx', header=None)
    df7 = pd.read_excel('Results_06062022_Dev7.xlsx', header=None)

    data = [df1, df2, df3, df4, df5, df6, df7]
    for df in data:
        df.rename({0: "x", 1: "y", 2: "c1", 3: "c2", 4: "c3", 5: "live"}, axis=1, inplace=True)

    viability1 = np.empty(len(data))
    viability2 = np.empty(len(data))

    for i in range(len(data)):
        viability1[i] = get_zone(data[i],3)
        viability2[i] = get_zone(data[i],4)

    dose_daun = np.array([400, 200, 100, 50, 25, 12.5, 6.25])
    dose_nel = np.array([400, 200, 100, 50, 25, 12.5, 6.25])
    dose_vcr = np.array([40, 20, 10, 5, 2.5, 1.25])

    # fit 
    hill1 = synergy.single.Hill()
    hill1.fit(dose_daun, viability1)

    hill2 = synergy.single.Hill()
    hill2.fit(dose_daun, viability2)

    print(hill1.get_parameters())
    print(hill2.get_parameters())

    dose_daun_fit = np.linspace(dose_daun.min(), dose_daun.max())
    E_daun_1 = hill1.E(dose_daun_fit)
    E_daun_2 = hill2.E(dose_daun_fit)
   
    # fit together
    dose_concat = np.concatenate((dose_daun, dose_daun))
    response_concat = np.concatenate((viability1, viability2))
    hill_all = synergy.single.Hill()
    hill_all.fit(dose_concat, response_concat)
    print(hill_all.get_parameters())
    E_all = hill_all.E(dose_daun_fit)

    # plotting
    fig, ax = plt.subplots()
    ax.scatter(dose_daun, viability1)
    ax.plot(dose_daun_fit, E_daun_1)
    ax.scatter(dose_daun, viability2)
    ax.plot(dose_daun_fit, E_daun_2)
    ax.plot(dose_daun_fit, E_all)
    #ax.set_xscale('log')
    ax.set_xlabel('Concentration [nM]')
    ax.set_ylabel('Viability')
    ax.set_ylim([0, 1])
    plt.show()

def zone_ratio(df):
    viability1, c1_1, c2_1, c3_1 = get_zone(df, 11)
    print(f"{(c1_1)/(c2_1)} : {1} ratio drug 1 to 2")


def main():
    df1 = pd.read_excel('Results_06032022_Dev1.xlsx', header=None)
    df2 = pd.read_excel('Results_06032022_Dev2.xlsx', header=None)
    df3 = pd.read_excel('Results_06032022_Dev3.xlsx', header=None)
    df4 = pd.read_excel('Results_06032022_Dev4.xlsx', header=None)
    df5 = pd.read_excel('Results_06032022_Dev5.xlsx', header=None)
    df6 = pd.read_excel('Results_06032022_Dev6.xlsx', header=None)
    df8 = pd.read_excel('Results_06032022_Dev8.xlsx', header=None)

    data = [df1, df2, df3, df4, df5, df6]
    for df in data:
        df.rename({0: "x", 1: "y", 2: "c1", 3: "c2", 4: "c3", 5: "live"}, axis=1, inplace=True)

    get_zone(df1, 11)
    zone_ratio(df1)

def main2():
    #test = Device.from_excel('Results_06032022_Dev1.xlsx', 20, 20, 40, 'nM')
    #test.print_summary()
    folder_name = 'device_test'
    drug1_idx = 1
    drug2_idx = 2

    test_stack = DeviceStack(folder_name)
    print(test_stack)

    # get max drug 1: vincristine on rep 1
    EC50_1, EC50_2, EC50_agg = test_stack.single_drug_fit(drug1_idx)
    drug1_EC50 = stats.median([EC50_1, EC50_2, EC50_agg])

    # get max drug 2: daunorubicin on rep 1
    EC50_1, EC50_2, EC50_agg = test_stack.single_drug_fit(drug2_idx)
    drug2_EC50 = stats.median([EC50_1, EC50_2, EC50_agg])

    target1, target2 = drug1_EC50/2, drug2_EC50/2
    print(target1)
    print(target2)

    # I now have to find a device in which the targets are matched up and can use a ratio
    # Translate the actual device concentrations into multiples of EC50?
    print(f"Normalized EC50 drug1: {np.around(test_stack.drug2_doses/drug1_EC50, 2)}")
    print(f"Normalized EC50 drug2: {np.around(test_stack.drug3_doses/drug2_EC50, 2)}")
    
    dose1_norm = test_stack.drug2_doses/drug1_EC50
    dose2_norm = test_stack.drug3_doses/drug2_EC50
    # the ratio is the normalized EC50 of drug 1 divided by the normalized EC50 of drug 2
    ratio = 1/np.mean(dose1_norm/dose2_norm)
    print(f"Ratio is: {ratio:.2f}")

    #print(test.get_zone_ratio(ratio))
    EC50_combo = test_stack.ratio_zone_fit(ratio, dose1_norm, dose2_norm)

    print(f"EC50 of combo is: {EC50_combo:.2f}")

    #print(test_stack.devices)
    #print(test_stack.drug1_name)
    #print(test_stack.drug1_doses)
    #print(test.get_zone_ratio(0.5))

def main3():
    #test = Device.from_excel('Results_06032022_Dev1.xlsx', 20, 20, 40, 'nM')
    #test.print_summary()
    folder_name = 'clustering_3'
    drug1_idx = 1
    drug2_idx = 2

    test_stack = DeviceStack(folder_name)
    print(test_stack)

    #test_stack.single_drug_fit(1)
    #test_stack.single_drug_fit(2)
    #test_stack.single_drug_fit(3)

    test_stack.combo_test(1,3)

if __name__ == '__main__':
    main3()
