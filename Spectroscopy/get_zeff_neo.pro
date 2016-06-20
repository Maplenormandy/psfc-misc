PRO loop_zeff_neo,shotmin,shotmax,dt=dt
	IF NOT keyword_set(dt) THEN dt=0.1

    numShots=shotmax-shotmin+1
    numTimes=floor((1.5-0.5)/dt)
    zeff_array=fltarr(numTimes,numShots)
    time_array=fltarr(numTimes,numShots)
    zeff_qfit_array=fltarr(numTimes,numShots)
    time_qfit_array=fltarr(numTimes,numShots)
    shot_array=make_array(numShots,/LONG)

    FOR i=0,numShots-1 DO BEGIN
        shot=shotmin+i
        print,shot
        shot_array[i]=shot
        zeff_neo,shot,zeff,time,dt=dt
        zeff_array[*,i]=zeff
        time_array[*,i]=time
        zeff_neo,shot,zeff_qfit,time_qfit,dt=dt,/qfit
        zeff_qfit_array[*,i]=zeff_qfit
        time_qfit_array[*,i]=time_qfit
    ENDFOR

    SAVE,shot_array,zeff_array,time_array,zeff_qfit_array,time_qfit_array,FILENAME='output.sav'
END
