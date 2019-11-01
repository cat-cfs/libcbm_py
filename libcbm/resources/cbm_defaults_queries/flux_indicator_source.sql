select flux_indicator_source.pool_id from flux_indicator
inner join flux_indicator_source on flux_indicator_source.flux_indicator_id = flux_indicator.id
where flux_indicator.id = ?