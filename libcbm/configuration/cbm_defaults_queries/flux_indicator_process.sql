select flux_indicator.id as flux_indicator_id, flux_process.id as flux_process_id from flux_indicator
inner join flux_process on flux_process.id = flux_indicator.flux_process_id