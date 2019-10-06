SELECT COUNT(*)
FROM
dmv
WHERE
dmv.record_type in ('SELECT record_type from dmv')
AND dmv.body_type in ('SELECT body_type from dmv WHERE body_type is not NULL')
AND dmv.fuel_type in ('SELECT fuel_type from dmv WHERE fuel_type is not NULL')
