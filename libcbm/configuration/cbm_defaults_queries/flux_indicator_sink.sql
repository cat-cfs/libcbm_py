select flux_indicator.id, flux_indicator_sink.pool_id from flux_indicator
inner join flux_indicator_sink on flux_indicator_sink.flux_indicator_id = flux_indicator.id