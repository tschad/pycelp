;;
;;  IDL File
;;
;;  Tom Schad 
;;
;; Notes:  Likely this could be made much faster.  It's slow for now. 
;;
;; History: 
;;    25 March 2021 :  original version 
;;    24 June 2021 : updating to Chianti v10 with expanded model for Si IX (590 levels)
;; ==========================================

ieqf = !xuvtop + '/ioneq/chianti.ioneq'
abf  = !xuvtop + '/abundance/sun_photospheric_2009_asplund.abund'

ions = ['fe_14','fe_11','fe_13','fe_13','si_10', 'si_9']
iontemps = 10.^[6.3,6.1,6.25,6.25,6.15,6.05]
wvl  = [   5304,   7894,  10749,  10800, 14305, 39293.]
nions = n_elements(ions)

;;
;; Temperature dependent contribution functions
;; Data for recreating Figure 2 of Schad & Dima (2020)
;;

logt = alog10([0.5:3.5:0.05]*1.e6)
nt   = n_elements(logt)
dens = 10.^(8.5)
rht  = 1.1

;goto,skip1

ch_int = dblarr(nt,nions)

for zz = 0,n_elements(ions)-1 do begin
  gname = ions[zz]
  ion2spectroscopic,gname,sname
  for tt = 0,nt-1 do begin
    print,' zz: ', zz,' tt: ', tt, ' of ',nt-1
    ch_synthetic,wvl[zz]-5.,wvl[zz]+5.,output = str1,/PHOTONS,$
                          SNGL_ION = ions[zz],err_msg = err_msg,$
                          ioneq_name = ieqf,LOGT_ISOTHERMAL = logt[tt], DENSITY = dens,$
                          LOGEM_ISOTHERMAL = 0.,/ALL,RPHOT = rht,NOPROT = 0,/VERBOSE

    if (err_msg eq '') then begin
       make_chianti_spec, str1,lambda,spectrum,/photons,abund_name = abf
       print,spectrum.lines[0].wvl
       ch_int[tt,zz] = spectrum.lines[0].int
    endif

  endfor
endfor

;;;str1.INT_UNITS = photons cm-2 sr-1 s-1

wnoz = where(ch_int NE 0.)
plot,logt,ch_int[*,0],/nodata,/ylog,yrange = minmax(ch_int[wnoz])
for zz = 0,n_elements(ions)-1 do oplot,logt,ch_int[*,zz]
save,filename = 'chianti_temp_contfnc_data.sav',logt,ch_int,ions,wvl,dens,rht

skip1:

;;
;; Density dependent contribution functions at the max temperatures defined
;; Densities between 5 and 15 dex
;; 27, 100, and all levels   (There are only 46 levels for Si IX)
;; protons and no protons
;; Height = 0.5
;; Data for recreating Figure 3 of Schad & Dima (2020)

rht = 1.5
dens = 10.^[5:15:0.2]
nd = n_elements(dens)

nlev   =  3 ;; 27, 100 and all
ch_int = DBLARR(2,nd,nlev,nions)  ;; (w/wo protons, ndens, (nlevels), (ions))

ions = ['fe_14','fe_11','fe_13','fe_13','si_10', 'si_9']

levels = intarr(3,nions) * 0
levels[0,*] = 27    ;; 27 levels
levels[1,*] = 100   ;; 100 levels if they exit
levels[2,*] = [739,996,749,749,204,590]  ;; all levels

for zz = 0,n_elements(ions)-1 do begin   ;; IONS / WAVELENGTHS
  gname = ions[zz]
  ion2spectroscopic,gname,sname
  cetemp = iontemps[zz]
  for mm = 0,nlev-1 do begin    ;; NUMBER OF LEVELS
    nlev_solve = levels[mm,zz]
    if (nlev_solve eq 0) then continue  ;; DO NOT LOOP IF NLEVELS = 0 (only for Si 9 which has less than 100)
    for dd = 0,nd-1 do begin    ;; LOOP OVER DENSITY
      for pp = 0,1 do begin     ;; LOOP OVER WITH AND WITHOUT PROTONS
        print,zz,mm,dd,pp,' number of levels: ',nlev_solve
        ch_synthetic_nlevels,wvl[zz]-5.,wvl[zz]+5.,output = str1,/PHOTONS,SNGL_ION = ions[zz],$
                             ioneq_name = ieqf,LOGT_ISOTHERMAL = alog10(cetemp), DENSITY = dens[dd],$
                             LOGEM_ISOTHERMAL = 0.,/ALL,nlevels = nlev_solve,RPHOT = rht,NOPROT = pp
        make_chianti_spec, str1,lambda,spectrum,/photons,abund_name = abf
        ch_int[pp,dd,mm,zz] = spectrum.lines[0].int
      endfor
    endfor
  endfor
endfor
print,spectrum.units
print,str1.int_units

save,filename ='./chianti_density_contfnc_data.sav',dens,levels,rht,iontemps,ch_int

STOP

END
