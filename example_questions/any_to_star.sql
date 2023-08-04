/*
 Example wide flat tables sharing attributes, which can be grouped to dimensions
 */

drop schema any_to_star cascade;
create schema any_to_star;

set search_path to any_to_star;

create table transactions(
  transaction_id int not null primary key,
  transaction_type int,
  transaction_type_name varchar,
  amount decimal(10, 2),
  customer_id int,
  customer_name varchar,
  customer_age int
);

create table loans(
  loan_id int not null primary key,
  loan_type int,
  loan_type_name varchar,
  amount decimal(10, 2),
  interest_rate decimal(10, 2),
  customer_id int,
  customer_name varchar,
  customer_age int
);
