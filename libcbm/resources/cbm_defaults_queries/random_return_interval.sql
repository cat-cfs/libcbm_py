select eco_boundary.id as eco_boundary_id,
random_return_interval.a_Nu,
random_return_interval.b_Nu,
random_return_interval.a_Lambda,
random_return_interval.b_Lambda
from eco_boundary inner join
random_return_interval on
eco_boundary.random_return_interval_id =
random_return_interval.id;