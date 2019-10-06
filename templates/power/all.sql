SELECT COUNT(*)
FROM
power
WHERE 'Xdate' < power.date
AND power.date <= 'Ydate'
AND 'Xtime' < power.time
AND power.time <= 'Ytime'
AND 'Xglobal_active_power' < power.global_active_power
AND power.global_active_power <= 'Yglobal_active_power'
AND 'Xglobal_reactive_power' < power.global_reactive_power
AND power.global_reactive_power <= 'Yglobal_reactive_power'
AND 'Xvoltage' < power.voltage
AND power.voltage <= 'Yvoltage'
AND 'Xglobal_intensity' < power.global_intensity
AND power.global_intensity <= 'Yglobal_intensity'
AND 'Xsub_metering_1' < power.sub_metering_1
AND power.sub_metering_1 <= 'Ysub_metering_1'
AND 'Xsub_metering_2' < power.sub_metering_2
AND power.sub_metering_2 <= 'Ysub_metering_2'
AND 'Xsub_metering_3' < power.sub_metering_3
AND power.sub_metering_3 <= 'Ysub_metering_3'
