SELECT COUNT(*)
FROM
dmv
WHERE
dmv.registration_class in ('SELECT registration_class from dmv')
AND dmv.fuel_type in ('SELECT fuel_type from dmv WHERE fuel_type is not NULL')
AND dmv.reg_valid_date in ('SELECT reg_valid_date from dmv WHERE reg_valid_date is not NULL')
AND dmv.scofflaw_indicator in ('SELECT distinct dmv.scofflaw_indicator from dmv')
