PRO loop_zeff_neo,shotmin,shotmax,dt=dt
    IF NOT keyword_set(dt) THEN dt=0.06

    numShots=2
    numTimes=16
    zeff_array=fltarr(numTimes,numShots)
    time_array=fltarr(numTimes,numShots)
    zeff_qfit_array=fltarr(numTimes,numShots)
    time_qfit_array=fltarr(numTimes,numShots)
    shot_array=[1120216017L, 1120216030L]

    FOR i=0,1 DO BEGIN
        shot=shot_array[i]
        print,shot
        zeff_neo,shot,zeff,time,dt=dt,n_zeff=20
        zeff_array[*,i]=zeff
        time_array[*,i]=time
        zeff_neo,shot,zeff_qfit,time_qfit,dt=dt,n_zeff=20,/qfit

        zeff_qfit_array[*,i]=zeff_qfit
        time_qfit_array[*,i]=time_qfit
    ENDFOR

    SAVE,shot_array,zeff_array,time_array,zeff_qfit_array,time_qfit_array,FILENAME='output.sav'
END
