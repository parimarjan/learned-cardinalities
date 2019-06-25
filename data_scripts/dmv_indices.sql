CREATE INDEX dmv3 ON dmv (record_type,registration_class,state,body_type,fuel_type,reg_valid_date,color);
CREATE INDEX dmv_scofflaw ON dmv (record_type,registration_class,state,body_type,fuel_type,reg_valid_date,color,suspension_indicator);
CREATE INDEX dmv_suspension ON dmv (record_type,registration_class,state,body_type,fuel_type,reg_valid_date,color,suspension_indicator);
CREATE INDEX dmv_revoc ON dmv (record_type,registration_class,state,body_type,fuel_type,reg_valid_date,color,revocation_indicator);

