select pool.id, pool.code 
from dom_pool 
left join pool 
on dom_pool.pool_id  is not NULL 
and pool.id > 0
order by pool.id;
