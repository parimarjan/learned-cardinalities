SELECT COUNT(*)
FROM
dmv
WHERE
dmv.record_type in ('SELECT record_type from dmv')
AND dmv.registration_class in ('SELECT registration_class from dmv')
AND dmv.state in ('SELECT state from dmv WHERE state is not null')
AND dmv.body_type in ('SELECT body_type from dmv WHERE body_type is not NULL')
AND dmv.fuel_type in ('SELECT fuel_type from dmv WHERE fuel_type is not NULL')
AND dmv.color in ('SELECT color from dmv WHERE color is not NULL')
AND dmv.reg_valid_date in ('SELECT reg_valid_date from dmv WHERE reg_valid_date is not NULL')
AND dmv.suspension_indicator in ('SELECT distinct dmv.suspension_indicator from dmv')
