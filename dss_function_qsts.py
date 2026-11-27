import numpy as np
import math
from scipy.sparse import lil_matrix
import scipy.sparse.linalg as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import networkx as nx
import csv
import random
 
def get_lines(dss):
    Line = []
    Switch = []
    lines = dss.Lines.First()
    while lines:
        datum = {}
        line = dss.Lines
        datum["name"] = line.Name()
        datum["bus1"] = line.Bus1()
        datum["bus2"] = line.Bus2()
        datum["switch_flag"] = dss.run_command('? Line.' + datum["name"] + '.Switch')
        datum["R1"] = line.R1()
        datum["X1"] = line.X1()
        datum["Rmatrix"] = line.RMatrix()
        datum["Xmatrix"] = line.XMatrix()
        if datum["switch_flag"] == 'False':
            datum["wires"] = dss.run_command('? Line.' + datum["name"] + '.Wires')
            datum["length"] = line.Length()
            datum["phases"] = line.Phases()
            datum["spacing"] = line.Spacing()
            datum["linecode"] = line.LineCode()
            datum["normAmp"] = dss.run_command('? Linecode.'+datum["linecode"]+'.normamps')
            Line.append(datum)
        else:
            Switch.append(datum)
        lines = dss.Lines.Next()
    return [Line, Switch]
 
def get_reactor(dss,circuit):
   
    Reactor = []
    circuit.SetActiveClass('Reactor')
    reactor_index = dss.ActiveClass.First()
    while reactor_index:
        datum = {}
        cktElement = dss.CktElement
        datum["Buses"] = cktElement.BusNames()
        datum["Name"] = cktElement.Name()
        datum["Open"] =cktElement.IsOpen(0,1)
        Reactor.append(datum)
        reactor_index = dss.ActiveClass.Next()
    return [Reactor]
 
def get_tran(dss):
    Transformer = []
    Trans = dss.Transformers.First()
    while Trans:
        datum = {}
        Tran = dss.Transformers
        datum["name"] = Tran.Name()
        datum["R"] = Tran.R()
        datum["xhl"] = Tran.Xhl()
        Tran.Wdg(1)
        datum["kV1"] = Tran.kV()
        Tran.Wdg(2)
        datum["kV2"] = Tran.kV()
        Tran.Wdg(1)
        datum["kVA1"] = Tran.kVA()
        Tran.Wdg(2)
        datum["kVA2"] = Tran.kVA()
        cktElement = dss.CktElement
        datum["Buses"] = cktElement.BusNames()
        Transformer.append(datum)
 
        Trans = dss.Transformers.Next()
    return [Transformer]
 
def get_transformer(dss,circuit):
    data = []
    circuit.SetActiveClass('Transformer')
    xfmr_index = dss.ActiveClass.First()
    while xfmr_index:
        dataline = []
        cktElement = dss.CktElement
        xfmr_name = cktElement.Name()
        buses = dss.run_command('? ' + xfmr_name + '.buses')
        conns = dss.run_command('? ' + xfmr_name + '.conns')
        kVs = dss.run_command('? ' + xfmr_name + '.kVs')
        kVAs = dss.run_command('? ' + xfmr_name + '.kVAs')
        phase = dss.run_command('? ' + xfmr_name + '.phases')
        loadloss = dss.run_command('? ' + xfmr_name + '.%loadloss')
        noloadloss = dss.run_command('? ' + xfmr_name + '.%noloadloss')
        Rs = dss.run_command('? ' + xfmr_name + '.%Rs')
        xhl = dss.run_command('? ' + xfmr_name + '.xhl')
        dataline = dict(name=xfmr_name,buses=buses,conns=conns,kVs=kVs,kVAs=kVAs,phase=phase,loadloss=loadloss,noloadloss=noloadloss,Rs=Rs,xhl=xhl)
        data.append(dataline)
        xfmr_index = dss.ActiveClass.Next()
    return data
 
def get_loads(dss, circuit):
    data = []
    load_flag = dss.Loads.First()
    total_load = 0
    while load_flag:
        load = dss.Loads
        datum = {
            "name": load.Name(),
            "kV": load.kV(),
            "kW": load.kW(),
            "PF": load.PF(),
            "Delta_conn": load.IsDelta()
        }
        indexCktElement = circuit.SetActiveElement("Load.%s" % datum["name"])
        cktElement = dss.CktElement
        bus = cktElement.BusNames()[0].split(".")
        datum["kVar"] = float(datum["kW"]) / float(datum["PF"]) * math.sqrt(1 - float(datum["PF"]) * float(datum["PF"]))
        datum["bus1"] = bus[0]
        datum["numPhases"] = len(bus[1:])
        datum["phases"] = bus[1:]
        if not datum["numPhases"]:
            datum["numPhases"] = 3
            datum["phases"] = ['1', '2', '3']
        datum["voltageMag"] = cktElement.VoltagesMagAng()[0]
        datum["voltageAng"] = cktElement.VoltagesMagAng()[1]
        s = dss.CktElement.Powers()
        datum["power"] = [sum(s[0:len(s):2]),sum(s[1:len(s):2])]
 
        data.append(datum)
        load_flag = dss.Loads.Next()
        total_load += datum["kW"]
    return [data, total_load]
 
def generate_PV(Load,totalLoadkW,target_penetration,outputfile):
    # need to use "get_loads" function to get "Load" and "totalLoadkW"
    pv_power = 0
    count = 1
    candidate = np.array(range(len(Load))).tolist()
    pv_dss = []
    while pv_power <= target_penetration/100*totalLoadkW:
        if not candidate:
            candidate = np.array(range(len(Load))).tolist()
        script = []
        load_index = random.randint(0,len(candidate)-1)
        load1 = Load[candidate[load_index]]
        
        for phase_no in load1["phases"]:
            pvname = load1["name"]+'_PV'+str(count)
            busname = load1["bus1"] + '.' + str(phase_no)
            kW = round(random.uniform(0,10),2) #randomly generate single PV system less than 10 kW
            #script = 'New Generator.' + pvname + ' bus1=' + busname + ' phases=1 kV=' + str(load1["kV"]/math.sqrt(load1["numPhases"])) + ' kW=' + str(kW) + ' kVA=' + str(kW/1.15) + ' pf=1 !yearly=PVshape_aggregated'       
            script = 'New PVSystem.' + pvname + ' bus1=' + busname + ' phases=1 kV=' + str(load1["kV"]/math.sqrt(load1["numPhases"])) + ' KVA=' + str(kW) + ' pmpp=' + str(kW) + ' model=1 pf=1 yearly=PVshape'
            count = count + 1
            pv_dss.append(script)
            pv_power = pv_power + kW
        candidate.remove(candidate[load_index])     
    
    file = open(outputfile,'w')
    for string in pv_dss:
        file.write(string + '\n')
    file.close()



def get_pvSystems(dss):
    data = []
    PV_flag = dss.PVsystems.First()
    while PV_flag:
        datum = {}
        PVname = dss.CktElement.Name()
        NumPhase = dss.CktElement.NumPhases()
        bus = dss.CktElement.BusNames()[0]
 
        PVkW = dss.run_command('? ' + PVname + '.Pmpp')
        PVpf = dss.run_command('? ' + PVname + '.pf')
        PVkVA = dss.run_command('? ' + PVname + '.kVA')
        PVkV = dss.run_command('? ' + PVname + '.kV')
 
        datum["name"] = PVname
        datum["bus"] = bus
        datum["Pmpp"] = PVkW
        datum["pf"] = PVpf
        datum["kV"] = PVkV
        datum["kVA"] = PVkVA
        datum["numPhase"] = NumPhase
        datum["power"] = dss.CktElement.Powers()[0:2*NumPhase]
 
        data.append(datum)
        PV_flag = dss.PVsystems.Next()
    return data
 

def get_Generator(dss):
    data = []
    gen_flag = dss.Generators.First()
    while gen_flag:
        datum = {}
        GENname = dss.CktElement.Name()
        NumPhase = dss.CktElement.NumPhases()
        bus = dss.CktElement.BusNames()[0]
        GENkW = dss.run_command('? ' + GENname + '.kW')
        GENpf = dss.run_command('? ' + GENname + '.pf')
        GENkVA = dss.run_command('? ' + GENname + '.kVA')
        GENkV = dss.run_command('? ' + GENname + '.kV')
        datum["name"] = GENname
        datum["bus"] = bus
        datum["kW"] = GENkW
        datum["pf"] = GENpf
        datum["kV"] = GENkV
        datum["kVA"] = GENkVA
        datum["numPhase"] = NumPhase
        #datum["power"] = dss.CktElement.Powers()[0:2*NumPhase]
        data.append(datum)
        gen_flag = dss.Generators.Next()
    return data
 

def get_capacitors(dss):
    data = []
    cap_flag = dss.Capacitors.First()
    while cap_flag:
        datum = {}
        capname = dss.CktElement.Name()
        NumPhase = dss.CktElement.NumPhases()
        bus = dss.CktElement.BusNames()[0]
        kvar = dss.run_command('? ' + capname + '.kVar')
        datum["name"] = capname
        temp = bus.split('.')
        datum["busname"] = temp[0]
        datum["busphase"] = temp[1:]
        if not datum["busphase"]:
            datum["busphase"] = ['1','2','3']
        datum["kVar"] = kvar
        datum["numPhase"] = NumPhase
        datum["power"] = dss.CktElement.Powers()[0:2 * NumPhase]
        data.append(datum)
        cap_flag = dss.Capacitors.Next()
    return data
 

def get_BusDistance(dss,circuit,AllNodeNames):
    Bus_Distance = []
    for node in AllNodeNames:
        circuit.SetActiveBus(node)
        Bus_Distance.append(dss.Bus.Distance())
    return Bus_Distance
       
 
def get_PVmaxP(dss,circuit,PVsystem):
    Pmax = []
    for PV in PVsystem:
        circuit.SetActiveElement(PV["name"])
        Pmax.append(-float(dss.CktElement.Powers()[0]))
    return Pmax
 

def get_PQnode(dss, circuit, Load, PVsystem, AllNodeNames,Capacitors):
    Pload = [0] * len(AllNodeNames)
    Qload = [0] * len(AllNodeNames)
    for ld in Load:
        for ii in range(len(ld['phases'])):
            name = ld['bus1'] + '.' + ld['phases'][ii]
            index = AllNodeNames.index(name.upper())
            circuit.SetActiveElement('Load.' + ld["name"])
            power = dss.CktElement.Powers()
            Pload[index] = power[2*ii]
            Qload[index] = power[2*ii+1]
 
    #PQ_load = np.matrix(np.array(Pload) + 1j * np.array(Qload)).transpose()
    PQ_load = np.array(Pload) + 1j * np.array(Qload)
 
    Ppv = [0] * len(AllNodeNames)
    Qpv = [0] * len(AllNodeNames)
    for PV in PVsystem:
        index = AllNodeNames.index(PV["bus"].upper())
        circuit.SetActiveElement(PV["name"])
        power = dss.CktElement.Powers()
        Ppv[index] = power[0]
        Qpv[index] = power[1]
 
    #PQ_PV = np.matrix(np.array(Ppv) + 1j * np.array(Qpv)).transpose()
    PQ_PV = -np.array(Ppv) - 1j * np.array(Qpv)
 
    Qcap = [0]*len(AllNodeNames)
    for cap in Capacitors:
        for ii in range(cap["numPhase"]):
            index = AllNodeNames.index(cap["busname"].upper()+'.'+cap["busphase"][ii])
            Qcap[index] = -cap["power"][2*ii-1]
 
    PQ_node = - PQ_load + PQ_PV + 1j*np.array(Qcap) # power injection
    return [PQ_load, PQ_PV, PQ_node, Qcap]
 
def get_subPower_byPhase(dss):
    dss.Lines.First()
    power = dss.CktElement.Powers()
    subpower = power[0:6:2]
    return subpower
 
def getTotalPowers(circuit, dss, type, names):
    d = [None] * len(names)
    count = 0
    for loadname in names:
        circuit.SetActiveElement(type+'.'+loadname)
        s = dss.CktElement.Powers()
        d[count] = [sum(s[0:len(s):2]),sum(s[1:len(s):2])]
        count = count + 1
    d = np.asarray(d)
    powers = sum(d)
    return powers
 
def getElemPower(circuit, dss, type, name):
    circuit.SetActiveElement(type+'.'+name)
    s = dss.CktElement.Powers()
    powers = [sum(s[0:int(len(s)/2)+2:2]),sum(s[1:int(len(s)/2)+2:2])]
    return powers
 
def getPVLoadPower(circuit, dss, type, name):
    circuit.SetActiveElement(type+'.'+name)
    s = dss.CktElement.Powers()
    powers = [sum(s[0:int(len(s)/2)+2:2]),sum(s[1:int(len(s)/2)+2:2])] # P added +2 in this formula
    return powers
 
def construct_Ymatrix(Ysparse, slack_no,totalnode_number,order_number):
    Ymatrix = np.array([[complex(0, 0)] * totalnode_number] * totalnode_number)
    file = open(Ysparse, 'r')
    G = []
    B = []
    count = 0
    for line in file:
        if count >= 4:
            temp = line.split('=')
            temp_order = temp[0]
            temp_value = temp[1]
            temp1 = temp_order.split(',')
            row_value = int(temp1[0].replace("[", ""))
            column_value = int(temp1[1].replace("]", ""))
            row_value = order_number[row_value-1]
            column_value = order_number[column_value-1]
            temp2 = temp_value.split('+')
            G.append(float(temp2[0]))
            B.append(float(temp2[1].replace("j", "")))
            Ymatrix[row_value, column_value] = complex(G[-1], B[-1])
            Ymatrix[column_value, row_value] = complex(G[-1], B[-1])         
        count = count + 1      
    file.close()
 
    Y00 = Ymatrix[0:slack_no, 0:slack_no]
    Y01 = Ymatrix[0:slack_no, slack_no:]
    Y10 = Ymatrix[slack_no:, 0:slack_no]
    Y11 = Ymatrix[slack_no:, slack_no:]
    Y11_sparse = lil_matrix(Y11)
    Y11_sparse = Y11_sparse.tocsr()
    a_sps = sparse.csc_matrix(Y11)
    lu_obj = sp.splu(a_sps)
    Y11_inv = lu_obj.solve(np.eye(totalnode_number-slack_no))
    return [Y00,Y01,Y10,Y11,Y11_sparse,Y11_inv]
 
def re_orgnaize_for_volt(V1_temp,AllNodeNames,NewNodeNames):
    V1 = [complex(0, 0)] * len(V1_temp)
    count = 0
    for node in NewNodeNames:
        index = AllNodeNames.index(node)
        print([count,index])
        V1[index] = V1_temp[count]
        count = count + 1
    return V1
 

def getCapsPos(dss,capNames):
    o = [None]*len(capNames)
    for i,cap in enumerate(capNames):
        x = dss.run_command('? capacitor.%(cap)s.states' % locals())
        o[i] = int(x[-2:-1])
    return o
 
def getRegsTap(dss,regNames):
    o = [None]*len(regNames)
    for i, name in enumerate(regNames):
        xfmr = dss.run_command('? regcontrol.%(name)s.transformer' % locals())
        res = dss.run_command('? transformer.%(xfmr)s.tap' % locals())
        o[i] = float(res)
    return o
 
def result(circuit,dss):
    res = {}
    res['AllVoltage'] = circuit.AllBusMagPu()
    temp = circuit.YNodeVArray()
    data = []
    for ii in range(int(len(temp)/2)):
        data.append(complex(temp[2*ii],temp[2*ii+1]))
    res['AllVolt_Yorder'] = data
    res['loss'] = circuit.Losses()
    res['totalPower'] = circuit.TotalPower() # power generated into the circuit
    loadname = dss.Loads.AllNames()
    res['totalLoadPower'] = getTotalPowers(circuit, dss, 'Load', loadname)
    capNames = dss.Capacitors.AllNames()
    if capNames:
        res['CapState'] = getCapsPos(dss,capNames)
    else:
        res['CapState'] = 'nan'
    regNames = dss.RegControls.AllNames()
    if regNames:
        res['RegTap'] = getRegsTap(dss,regNames)
    else:
        res['RegTap'] = 'nan'
       
    pvNames = dss.PVsystems.AllNames()
    dataP = np.zeros(len(pvNames))
    dataQ = np.zeros(len(pvNames))
    ii = 0
    sumP = 0
    sumQ = 0
    for pv in pvNames:
    #    circuit.SetActiveElement('PVsystem.'+pv)
        circuit.SetActiveElement('PVSystem.'+pv)
        tempPQ = dss.CktElement.Powers()
        dataP[ii] = sum(tempPQ[0:int(len(tempPQ)/2)+2:2])
        dataQ[ii] = sum(tempPQ[1:int(len(tempPQ)/2)+2:2])
        sumP = sumP + dataP[ii]
        sumQ = sumQ + dataQ[ii]
        ii = ii + 1
    res['PV_Poutput'] = dataP
    res['PV_Qoutput'] = dataQ
    res['totalPVpower'] = [sumP,sumQ]   
    return res
 
def get_Vbus(dss,circuit,busname):
    circuit.SetActiveBus(busname)
    voltage = dss.Bus.VMagAngle()
    Vmag = [ii/dss.Bus.kVBase()/1000 for ii in voltage[0:len(voltage):2]]
    return Vmag
   
def get_Vcomplex_Yorder(circuit,node_number):
    temp_Vbus = circuit.YNodeVArray()
    Vbus = [complex(0,0)]*node_number
    for ii in range(node_number):
        Vbus[ii] = complex(temp_Vbus[ii*2],temp_Vbus[ii*2+1])
    return Vbus
 
def get_VmagPU_Yorder(circuit,Vbase):
    temp = circuit.YNodeVArray()
    Vbus = []
    for ii in range(int(len(temp)/2)):
        Vbus.append(complex(temp[2*ii],temp[2*ii+1]))   
    Vbus_pu = list(map(lambda x: abs(x[0])/x[1],zip(Vbus,Vbase)))
    return Vbus_pu
 

def create_graph(dss, phase):  
    df = dss.utils.lines_to_dataframe()   
    G = nx.Graph()
    data = df[['Bus1', 'Bus2']].to_dict(orient="index")
    for name in data:
        line = data[name]
        if ".%s" % phase in line["Bus1"] and ".%s" % phase in line["Bus2"]:
            G.add_edge(line["Bus1"].split(".")[0], line["Bus2"].split(".")[0])
    pos = {}
    for name in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus("%s" % name)
        if phase in dss.Bus.Nodes():
            index = dss.Bus.Nodes().index(phase)
            re, im = dss.Bus.PuVoltage()[2*index:2*index+2]
            V = abs(complex(re, im))
            D = dss.Bus.Distance()
            pos[dss.Bus.Name()] = (D, V)          
    return G, pos
 
def plot_profile(dss, phase):  
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    ax = axs      
    ncolor = ['k', 'r', 'b']
    if phase==1:
         G, pos = create_graph(dss, 1)
         nx.draw_networkx_nodes(G, pos, ax=ax, node_size=10)
         nx.draw_networkx_edges(G, pos, ax=ax, node_size=10)
         ax.set_title("Voltage profile plot for phase A")
        
    elif phase==2:
         G, pos = create_graph(dss, 2)
         nx.draw_networkx_nodes(G, pos, ax=ax, node_size=10)
         nx.draw_networkx_edges(G, pos, ax=ax, node_size=10)
         ax.set_title("Voltage profile plot for phase B")
    elif phase==3:
         G, pos = create_graph(dss, 3)
         nx.draw_networkx_nodes(G, pos, ax=ax, node_size=10)
         nx.draw_networkx_edges(G, pos, ax=ax, node_size=10)
         ax.set_title("Voltage profile plot for phase C")
    else:        
         for ph in range(3):
             G, pos = create_graph(dss, ph+1)
             color=['blue', 'green', 'red']
             nx.draw_networkx_nodes(G, pos, ax=ax, node_size=10,node_color=color[ph])
             nx.draw_networkx_edges(G, pos, ax=ax, node_size=10, edge_color=color[ph])
         ax.set_title("Voltage profile plot for all phases")
                 
    ax.grid()
    ax.set_ylabel("Voltage in p.u.")
    ax.set_xlabel("Distances in km")
    plt.show()
   
def engo_setpoint_update(dss,circuit,dfEngos,setpoint,engo_fname):
    engoname = []
    for engonum in range(len(dfEngos)):
        engoname.append('ENGO'+str(engonum+1))
       
    f = open(engo_fname, 'w')
    for engo in range(len(dfEngos)):
        #for engo in range(3):
        f.write('New Capacitor.%s Phases=1 Bus1=%s kvar=10 numsteps=10 kv=%s\n' % (engoname[engo], dfEngos.loc[engo, 'bus'], str(round(dfEngos.loc[engo, 'kV'],3))))
        f.write('New capcontrol.%s Element=Transformer.%s Terminal=2 capacitor=%s ctr=1 ptr=1  EventLog=Yes \n\
                 ~ usermodel="C:\Program Files\OpenDSS\\x64\ENGOCapControl_12345_sec.dll" \n\
                 ~ userdata=(ENABLE=Y Vnom=%s Vsp_120b =%s Vband_120b =1 )\n\n' % (dfEngos.loc[engo, 'Name'], dfEngos.loc[engo, 'Transformer'], dfEngos.loc[engo, 'Name'], str(int(float(dfEngos.loc[engo, 'kV'])*1000)), str(setpoint)))
    f.close()
   
 
def getElemCurrents(circuit, dss, type, name):
    circuit.SetActiveElement(type+'.'+name)
    s = dss.CktElement.CurrentsMagAng()
    magIbyPh = s[0:len(s):2][:3]
 
    return magIbyPh
 
def get3phLinePower(circuit, dss, type, name):
    circuit.SetActiveElement(type+'.'+name)
    s = dss.CktElement.Powers()
    lens = int(len(s)/2)
    powers = [sum(s[0:lens:2]),sum(s[1:lens:2])]
    return powers
 
def getElemPower_byPhase(circuit, dss, type, name):
    circuit.SetActiveElement(type+'.'+name)
    s = dss.CktElement.Powers()   
    powers_P = s[0:int(len(s)/2):2]
    powers_Q = s[1:int(len(s)/2):2]
    powers_PQ = [powers_P,powers_Q]
    return powers_PQ