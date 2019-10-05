SELECT COUNT(*)
FROM
dmv
WHERE
dmv.state in ('SELECT state from dmv WHERE state is not null')
AND dmv.body_type in ('SELECT body_type from dmv WHERE body_type is not NULL')
AND dmv.fuel_type in ('SELECT fuel_type from dmv WHERE fuel_type is not NULL')
AND dmv.color in ('SELECT color from dmv WHERE color is not NULL')
AND dmv.suspension_indicator in ('SELECT distinct dmv.suspension_indicator from dmv')
