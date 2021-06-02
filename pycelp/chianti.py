
import numpy as np
import os
import re

#__all__ = ['getChiantiDir','getIonZ']

def getChiantiDir():
    try:
        chianti_dbase_root = os.environ['XUVTOP']
    except:
        raise NameError('Environmental variable for Chianti Database not defined')
    ## FOR A DEBUG MODE: print('Chianti database location: ',chianti_dbase_root)
    with open(os.path.join(chianti_dbase_root, 'VERSION'), 'r') as f:
        lines = f.readlines()
        chianti_version = lines[0].strip()
    ## print('Chianti version: ',chianti_version)
    return chianti_dbase_root,chianti_version

def getIonZ(ion_name):
    '''
    Get Atomic Number of the Element given an ion name, e.g. 'fe_13'
    '''
    element, ion = ion_name.split('_')
    zlabl=np.array(['H','He','Li','Be','B','C','N','O','F','Ne','Na',
       'Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr',
       'Mn','Fe','Co','Ni','Cu','Zn'])
    zlabl_upp = np.char.upper(zlabl)
    wion = np.where(element.upper() == zlabl_upp)[0]
    ionz = wion[0] + 1
    return ionz

def getAtomicWeight(element):
    """
    Get atomic weight of element
    """
    elem = np.array(['h','he','li','be','b','c','n','o','f',
        'ne','na','mg','al','si','p','s','cl','ar','k',
        'ca','sc','ti','v','cr','mn','fe'])
    wgts = np.array([1.008,4.0026,6.94,9.012,10.81,12.011,14.007,
        15.999,18.998,20.1797,22.9897,24.305,26.9815,
        28.085,30.97376,32.06,35.45,39.948,39.0983,
        40.078,44.955,47.867,50.94,51.996,54.938,55.845])
    wem = np.where(elem == element)
    return wgts[wem[0]][0]

def readFile(filename):
    print(' reading: ',filename)
    with open(filename, 'r') as f:
        lines = f.readlines()

    for i,line in enumerate(lines):
        if line.strip() == '-1' or r'%file' in line:  # sometimes the first -1 is missing
            i_start = i if r'%file' in line else i+1
            data = lines[0:i]
            footer = lines[i_start:len(lines)]
            break

    reference = [ln.strip() for ln in footer]
    return data,reference

def convertConfig(conf):
    """
    Translated from convert_config.pro in Chianti IDL version
    """
    conf = conf.strip()  ## remove leading and trailing spaces
    conf = conf.replace('.',' ')  ## replace '.' with spaces
    conf = conf.replace('(',' (')  ## insert a space where parentheses are
    conf = conf.replace(')',') ')
    index1 = [i for i in range(len(conf)) if conf.startswith('(', i)]
    index2 = [i for i in range(len(conf)) if conf.startswith(')', i)]
    assert len(index1) == len(index2), ' Number of open and closed parentheses must match'

    if len(index1) == 0:
        conf = conf.lower()  ## if there are no parentheses, then lowercase everything
        conf_latex = conf  ## bring along a copy of the latex version as well
    else:
        ## split string into components
        indx = [None] * (len(index1)*2+2)
        indx[0] = 0
        indx[1:-1:2] = index1
        index2n = [i+1 for i in index2]  ## add one to the index2 to include closing parenthesis
        indx[2:-1:2] = index2n
        indx[-1] = len(conf)
        nparts = len(indx)-1
        parts = [None] * nparts
        for n in range(nparts):
            parts[n] = conf[indx[n]:indx[n+1]]
        ## create list to hold LATEX version
        parts_latex = [None] * nparts
        ## make everything outside of parenthesis lowercase
        for n in range(0,nparts,2):
            parts[n] = parts[n].lower()
            parts_latex[n] = parts[n].lower()
        ## inside the parthensis if first character after the '(' is an integer
        ## then proceee and uppercase what is inside
        for n in range(1,nparts,2):
            if parts[n][1].isnumeric():
                parts[n] = parts[n].upper()
                parts_latex[n] = parts[n][0] + '$^' + parts[n][1] + '$' + parts[n][2:]
            else:
                parts[n] = parts[n]
                parts_latex[n] = parts[n]
        ## print parts
        conf = ''
        conf_latex = ''
        for n in range(nparts):
            conf += parts[n]
            conf_latex += parts_latex[n]

    ##
    spdf   = ['s', 'p', 'd', 'f', 'g','h','i','k']
    n_spdf = len(spdf)
    parity = 0
    for p in range(n_spdf):
        index1 = [i for i in range(len(conf)) if conf.startswith(spdf[p], i)]
        index2 = [i for i in range(len(conf_latex)) if conf.startswith(spdf[p], i)]
        for ii,indx in enumerate(index1):
            s1,s2,s3 = '','',''
            if (indx+1)<len(conf): s1 = conf[indx+1]
            if (indx+2)<len(conf): s2 = conf[indx+2]
            if (indx+3)<len(conf): s3 = conf[indx+3]
            s1n,s2n,s3n = False,False,False
            if s1.isnumeric(): s1n = True
            if s2.isnumeric(): s2n = True
            if s3.isnumeric(): s3n = True
            if s1n and (s2 == ' ' or s2 == ''):  ## E.g., '3s2 ' or '3s2'
                occup_str, occup,j,jlatex = s1, np.int(s1), indx+2, index2[ii]+2
            elif not s1n:  ## E.g., '3s '
                occup_str, occup,j,jlatex = '',1,indx+1, index2[ii]+1
            elif s1n and s2n and (s3 == ' ' or s3 == ''): # E.g., '3d10 4s'
                occup_str, occup,j,jlatex = s1+s2, np.int(s1+s2), indx+3, index2[ii]+3
            elif s1n and s2n and s3n: # E.g., '3d104s'
                occup_str, occup,j,jlatex = s1+s2, np.int(s1+s2), indx+2, index2[ii]+2
            elif s1n and s2n and not s3n: # E.g., '3s22p'
                occup_str, occup,j,jlatex = s1,np.int(s1),indx+2, index2[ii]+2
            elif s1n and not s2n and (s2 != ' '): # E.g., '3s2p'
                occup_str, occup,j,jlatex = '',1,indx+1, index2[ii]+1
            else: # E.g., '4s'
                occup_str, occup,j,jlatex = '',1,indx+1, index2[ii]+1

            if (p%2): parity = parity + occup   ## non odd orbitals do not contribute to parity
            if occup == 1: occup_str = ''
            if occup_str != '':
                add_str = '$^{' + occup_str + '}$'
                conf_latex = conf_latex[0:index2[ii]+1] + add_str + conf_latex[jlatex:]
                add_str = occup_str
                conf = conf[0:indx+1] + add_str + conf[j:]

    if (parity%2):
        parity = 1
    else:
        parity = 0

    return conf,conf_latex,parity

def elvlcRead(ion_name):
    '''
    ion_name, e.g.  'fe_13'
    '''
    chianti_dbase_root,chianti_version = getChiantiDir()
    element, ion = ion_name.split('_')
    ionZ = getIonZ(ion_name)
    filename = chianti_dbase_root + element + os.path.sep + ion_name + os.path.sep + ion_name + '.elvlc'
    data,reference = readFile(filename)

    nlvl = len(data)

    ## setup data variables
    index  = [None] * nlvl   ## level index
    conf   = [None] * nlvl   ## configuration
    label  = [None] * nlvl   ## level label
    mult   = [None] * nlvl   ## multiplicity, 2s+1
    l_sym  = [None] * nlvl   ## orbital angular momentum
    j      = [None] * nlvl   ## total angular momentum
    obs_energy = [None] * nlvl   ## observed energy
    theory_energy = [None] * nlvl   ## theoretical energy

    elvlcFormat = '(I7,A30,A5,I5,A5,F5.1,F15.3,F15.3)'
    widFormat = [0,7,30,5,5,5,5,15,15]
    widC = np.cumsum(widFormat)

    for i,ln in enumerate(data):
        chunks = [ln[widC[j]:widC[j+1]] for j in range(len(widC)-1)]
        index[i] = np.int(chunks[0])
        conf[i]  = np.str(chunks[1]).strip()
        label[i] = np.str(chunks[2]).strip()
        mult[i]  = np.int(chunks[3])
        l_sym[i] = np.str(chunks[4]).strip()
        j[i]     =  np.float(chunks[5])
        obs_energy[i] = np.float(chunks[6])
        theory_energy[i] = np.float(chunks[7])

    ## best energy
    energy = [None] * nlvl
    for i in range(nlvl):
        energy[i] = obs_energy[i]
        if obs_energy[i] == -1: energy[i] = theory_energy[i]

    conf_index = [None] * nlvl
    conf_latex = [None] * nlvl
    parity = [None] * nlvl
    parity_str = [None] * nlvl

    uconf, uconf_rindx = np.unique(conf,return_inverse=True)
    cfg_indx = 1
    par_str = ['e','o']
    for i in range(nlvl):
        if conf_index[i] == None:
            uconf_i = uconf[uconf_rindx[i]]
            wi = [k for k,s in enumerate(conf) if uconf_i in s]
            c1,clatex,par = convertConfig(uconf_i)
            for ww in wi:
                conf_index[ww] = cfg_indx
                conf_latex[ww] = clatex
                parity[ww] = par
                parity_str[ww] = par_str[par]
            cfg_indx += 1

    spd = ['S','P','D','F','G','H','I','K','L','M','N','O','Q','R','T','U','V','W','X','Y']

    ## get orbital angular momentum quantum number (l)
    l = [None] * nlvl
    for i in range(len(spd)):
        wi = [k for k,lsym in enumerate(l_sym) if lsym == spd[i]]
        for ww in wi: l[ww] = i

    ## spin quantum number
    s = (np.asarray(mult,dtype=np.float)-1.)/2.
    ## degeneracy
    weight = 2.*np.asarray(j) + 1.

    term = [None] * nlvl
    term_latex = [None] * nlvl
    j_str = [None] * nlvl
    level = [None] * nlvl
    level_latex = [None] * nlvl
    full_level = [None] * nlvl
    full_level_latex = [None] * nlvl
    for i in range(nlvl):
        term[i] = str(mult[i]) + l_sym[i]
        term_latex[i] = '$^' + str(mult[i]) + '$' + l_sym[i]
        if (np.int(j[i]) == j[i]):
            j_str[i] = str(np.int(j[i]))
        else:
            j_str[i] = str(np.int(j[i]*2)) + '/2'
        level[i] = term[i] + j_str[i]
        level_latex[i] = term_latex[i] + '$_{' + j_str[i] + '}$'
        full_level[i] = conf[i] + ' ' + level[i]
        full_level_latex[i] = conf_latex[i] + ' ' + level_latex[i]

    result =   {"ion_name": ion_name,
                "ion_z":ionZ,
                "ion_n":ion,
                "filename":filename,
                "version":chianti_version,
                "reference":reference,
                "index":np.array(index),
                "conf":np.array(conf),
                "conf_latex":np.array(conf_latex),
                "conf_index":np.array(conf_index),
                "term":np.array(term),
                "term_latex":np.array(term_latex),
                "level":np.array(level),
                "level_latex":np.array(level_latex),
                "full_level":np.array(full_level),
                "full_level_latex":np.array(full_level_latex),
                "label":np.array(label),
                "mult":np.array(mult),
                "s":np.array(s),
                "l":np.array(l),
                "l_sym":np.array(l_sym),
                "j":np.array(j),
                "j_str":np.array(j_str),
                "parity":np.array(parity),
                "parity_str":np.array(parity_str),
                "weight":np.array(weight),
                "obs_energy":np.array(obs_energy),
                "theory_energy":np.array(theory_energy),
                "energy":np.array(energy),
                "energy_units": 'cm^-1'}

    return result

def wgfaRead(ion_name):
    """"
    Reads the wgfa file for a given ion from the chianti database.
    Assumes $XUVTOP is defined as environmental variable

    Parameters
    ----------
    ion_name : `str`
        Name of ion, e.g. 'fe_13'
    """
    chianti_dbase_root,chianti_version = getChiantiDir()
    element, ion = ion_name.split('_')
    filename = chianti_dbase_root + element + os.path.sep + ion_name + os.path.sep + ion_name + '.wgfa'
    data,reference = readFile(filename)

    n = len(data)
    lower_level_index = np.zeros(n,dtype = np.int)  ## lower level index
    upper_level_index = np.zeros(n,dtype = np.int)  ## upper level index
    wavelength  = np.zeros(n,dtype = np.float)  ## transition wavelength
    gf = np.zeros(n,dtype = np.float)  ## oscillator strength
    A_einstein = np.zeros(n,dtype = np.float)  ## radiative decay rate

    wgfaFormat = '(2i5,f15.3,2e15.3)'
    for i,ln in enumerate(data):
        lower_level_index[i] = np.int(ln[0:5])
        upper_level_index[i] = np.int(ln[5:10])
        wavelength[i] = np.float(ln[10:25])
        gf[i] = np.float(ln[25:40])
        A_einstein[i] = np.float(ln[40:55])

    result = {"lower_level_index":np.array(lower_level_index),
              "upper_level_index":np.array(upper_level_index),
              "wavelength":np.array(wavelength),
              "gf":np.array(gf),
              "A_einstein":np.array(A_einstein),
              "filename":filename,
              "version":chianti_version,
              "reference":reference}

    return result

def scupsRead(ion_name):
    """
    Reads scups files
    """
    chianti_dbase_root,chianti_version = getChiantiDir()
    element, ion = ion_name.split('_')
    filename = chianti_dbase_root + element + os.path.sep + ion_name + os.path.sep + ion_name + '.scups'
    data,reference = readFile(filename)

    n = len(data)//3   ## data for each transition spread over three lines

    lower_level_index = []
    upper_level_index = []
    delta_energy = []
    gf = []
    high_t_limit = []
    n_t = []
    bt_type = []
    bt_c = []
    bt_t = []
    bt_upsilon = []

    ## fortran formating for line 1 of 3
    scupsFormat = '(I7,I7,E12.3,E12.3,E12.3,I5,I3,E12.3)'
    ## I believe the I3 format is incorrect
    widFormat = [0,7,7,12,12,12,5,5,12]
    widC = np.cumsum(widFormat)

    for i in range(n):
        ln1 = data[i*3+0]
        ln2 = data[i*3+1]
        ln3 = data[i*3+2]
        chunks = [ln1[widC[j]:widC[j+1]] for j in range(len(widC)-1)]
        lower_level_index.append(np.int(chunks[0]))
        upper_level_index.append(np.int(chunks[1]))
        delta_energy.append(np.float(chunks[2]))
        gf.append(np.float(chunks[3]))
        high_t_limit.append(np.float(chunks[4]))
        n_t.append(np.int(chunks[5]))
        bt_type.append(np.int(chunks[6]))
        bt_c.append(np.float(chunks[7]))
        bt_t.append(np.array(ln2.split(),dtype = np.float))
        bt_upsilon.append(np.array(ln3.split(),dtype = np.float))

    ntrans = len(bt_t)
    nt_max = np.max(n_t)
    bt_t_arr = np.zeros((ntrans,nt_max))
    bt_u_arr = np.zeros((ntrans,nt_max))
    for n in range(ntrans):
        bt_t_arr[n,0:n_t[n]] = bt_t[n]
        bt_u_arr[n,0:n_t[n]] = bt_upsilon[n]

    result = {"lower_level_index":np.array(lower_level_index),
            "upper_level_index":np.array(upper_level_index),
            "delta_energy":np.array(delta_energy),
            "gf":np.array(gf),
            "high_t_limit":np.array(high_t_limit),
            "n_t":np.array(n_t),
            "bt_type":np.array(bt_type),
            "bt_c":np.array(bt_c),
            "bt_t":bt_t_arr,
            "bt_upsilon":bt_u_arr,
            "filename":filename,
            "version":chianti_version,
            "reference":reference}

    return result

def splupsRead(ion_name):
    """
    Reads splups proton rate files
    """
    chianti_dbase_root,chianti_version = getChiantiDir()
    element, ion = ion_name.split('_')
    filename = chianti_dbase_root + element + os.path.sep + ion_name + os.path.sep + ion_name + '.psplups'
    data,reference = readFile(filename)

    n = len(data)

    lower_level_index = []
    upper_level_index = []
    t_type = []
    gf = []
    delta_energy = []
    bt_c = []
    nspl = []
    bt_t = []
    bt_upsilon = []

    for i in range(n):
        ln = data[i]
        ## figure out number of spline points by subtracting
        ## length of first 6 elements from line and divide by
        ## length of spline value
        npts = int((len(ln) -3*3-3*10)/10)
        widFormat = [0,3,3,3,10,10,10]
        for ss in range(npts): widFormat.append(10)
        widC = np.cumsum(widFormat)
        chunks = [ln[widC[j]:widC[j+1]] for j in range(len(widC)-1)]
        lower_level_index.append(int(chunks[0]))
        upper_level_index.append(int(chunks[1]))
        t_type.append(int(chunks[2]))
        gf.append(np.float(chunks[3]))
        delta_energy.append(np.float(chunks[4]))
        bt_c.append(np.float(chunks[5]))
        nspl.append(npts)
        bt_t.append(np.linspace(0,1,npts))
        bt_upsilon.append(np.array(chunks[6:],dtype=np.float))

    result = {"lower_level_index":np.array(lower_level_index),
            "upper_level_index":np.array(upper_level_index),
            "bt_type":np.array(t_type),
            "gf":np.array(gf),
            "delta_energy":np.array(delta_energy),
            "bt_c":np.array(bt_c),
            "n_t":np.array(nspl),
            "bt_t":np.array(bt_t),
            "bt_upsilon":np.array(bt_upsilon),
            "filename":filename,
            "version":chianti_version,
            "reference":reference}

    return result


def abundRead(filename):

    chianti_dbase_root,chianti_version = getChiantiDir()
    filename = chianti_dbase_root + 'abundance' + os.path.sep + 'sun_photospheric_2009_asplund.abund'
    print(' testing default file:',filename)

    data,reference = readFile(filename)

    n = len(data)
    abund_z   = np.zeros(n,dtype=np.int)
    abund_val = np.zeros(n,dtype=np.float)

    for i,ln in enumerate(data):
        lns = ln.split()
        abund_z[i] = np.int(lns[0])
        abund_val[i] = np.float(lns[1])

    result = {"abund_z":abund_z,
        "abund_val":abund_val,
        "filename":filename,
        "version":chianti_version,
        "reference":reference}

    return result


def ioneqRead(filename):
    """
    Reads an ioneq file
    ionization equilibrium values less then minIoneq are returns as zeros
    Returns
    -------
    {'temp','ionfrac','filename','version','reference'} : `dict`
        Ionization equilibrium values and the reference to the literature
    """
    chianti_dbase_root,chianti_version = getChiantiDir()
    filename = chianti_dbase_root + 'ioneq' + os.path.sep + 'chianti.ioneq'
    print(' testing default file:',filename)
    data,reference = readFile(filename)

    n = len(data)
    nt = np.int(data[0].split()[0])  ## number of temperatures
    nz = np.int(data[0].split()[1])  ## number of elements

    ## get temperatures
    t = np.asarray(data[1].split()).astype(np.float)

    ioneq = np.zeros((nt,nz,nz+1))

    for i,ln in enumerate(data[2:]):
        lns = ln.split()
        z1,ion1 = np.int(lns[0]),np.int(lns[1])
        ioneq[:,z1-1,ion1-1] = np.asarray(lns[2:]).astype(np.float)

    result = {"temp":t,
            "ionfrac":ioneq,
            "filename":filename,
            "version":chianti_version,
            "reference":reference}

    return result

def limit_levels(input,nlevels,type = None):
    """

    """

    if (type == None):
        print('A type must be speficed for the limit levels routine')
        raise

    if (type == 'elvl'):

        nlev_input = len(input['index'])
        if (nlev_input < nlevels):
            print(' --- Number of requested levels exceeds those in database')
            print(' --- Using all levels')
            return input

        keys = 'index','conf','conf_latex','conf_index','term','term_latex', \
        'level','level_latex','full_level','full_level_latex','label','mult', \
        's','l','l_sym','j','j_str','parity','parity_str','weight', \
        'obs_energy','theory_energy','energy'

        for k in keys:
            input[k] = input[k][0:nlevels]

        return input

    if (type == 'wgfa'):

        lowlev = input['lower_level_index']
        upplev = input['upper_level_index']
        wg = (lowlev < nlevels+1) * (upplev < nlevels+1)

        keys = 'lower_level_index','upper_level_index','wavelength', \
        'gf','A_einstein'

        for k in keys:
            input[k] = input[k][wg]

        return input

    if (type == 'scups'):

        lowlev = input['lower_level_index']
        upplev = input['upper_level_index']
        wg = (lowlev < nlevels+1) * (upplev < nlevels+1)

        keys = 'lower_level_index','upper_level_index','delta_energy', \
        'gf','high_t_limit','n_t','bt_type','bt_c','bt_t','bt_upsilon'

        for k in keys:
            input[k] = input[k][wg]

        return input

    if (type == 'splups'):

        lowlev = input['lower_level_index']
        upplev = input['upper_level_index']
        wg = (lowlev < nlevels+1) * (upplev < nlevels+1)

        keys = 'lower_level_index','upper_level_index','bt_type', \
        'gf','delta_energy','bt_c','n_t','bt_t','bt_upsilon'

        for k in keys:
            input[k] = input[k][wg]

        return input


if __name__ == "__main__":

    print(' this is a module of chianti IO routines')
