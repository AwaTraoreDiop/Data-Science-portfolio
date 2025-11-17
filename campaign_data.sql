-- Customers overview in the dataset ?
drop table #overview
select count(distinct customer_id) customer
, AVG(round(datediff(month, birth_date,  event_date)/12, 0)) as age
, AVG(averageBasket) as AOV
, max(averageBasket) max_aov
, min(averageBasket) min_aov
into #overview
from #campaign_dataset

--select * from #overview
--Analysis :
--10k customers recieved communication from CRM. 
-- They are most likely middle aged 
-- in average, they spent 549 within the last 12 months

-- what is the second higher basket ? 
select max(averageBasket) as aov
from #campaign_dataset
where averageBasket<(select max(averageBasket) as aov from #campaign_dataset)


-- gender distribution 
select gender
, count(distinct customer_id) as cust
from #campaign_dataset
group by gender
order by cust 

--Analysis :
-- 1/3 of customers do not have an identified gender
-- there are as much men as women 

-- age cluster
drop table #customer
select customer_id
, AVG(round(datediff(month, birth_date,  event_date)/12, 0)) as age
into #customer
from #campaign_dataset as A
group by customer_id

;with age_group AS (
	select *
	, case when age<=18 then '<=18'
		when age<20 then '[16-20[' 
		when age<30 then '[20-30['   
		when age<40 then '[30-40['   
		when age<50 then '[40-50['   
		when age<60 then '[50-60['
		when age<70 then '[60-70['
		when age>=70 then '[70 +['  
		end as age_group
	from #customer
) 

select age_group
, count(distinct customer_id) as cust
from age_group
group by age_group
order by age_group 

-- How many targets by campaign? 

select campaign
, count(distinct customer_id) as cust
, AVG(averageBasket) as AOV
from #campaign_dataset
group by campaign
order by cust desc

--Analysis : most customers were targeted via Fashion week

-- targets over weeks : histograms to be created to observe the seasonality, should it exists

select datepart(iso_week, event_date) as weeks 
, count(distinct customer_id) as cust
, sum(averageBasket) as amount
from #campaign_dataset
group by datepart(iso_week, event_date)
order by weeks

-- ... to be continued
