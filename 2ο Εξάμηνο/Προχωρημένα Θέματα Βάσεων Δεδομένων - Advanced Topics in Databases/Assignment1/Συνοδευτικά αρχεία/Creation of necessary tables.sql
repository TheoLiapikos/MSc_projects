

-- Table: public.people

-- DROP TABLE public.people;

CREATE TABLE public.people
(
    name character varying(20) COLLATE pg_catalog."default" NOT NULL,
    parcel_id integer,
    CONSTRAINT people_pkey PRIMARY KEY (name),
    CONSTRAINT parcelid FOREIGN KEY (parcel_id)
        REFERENCES public.parcels ("OID") MATCH SIMPLE
        ON UPDATE CASCADE
        ON DELETE CASCADE
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;


-- Table: public.plants

-- DROP TABLE public.plants;

CREATE TABLE public.plants
(
    name character varying(20) COLLATE pg_catalog."default" NOT NULL,
    season character varying(10) COLLATE pg_catalog."default",
    CONSTRAINT plants_pkey PRIMARY KEY (name)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;


-- Table: public.annual_performances

-- DROP TABLE public.annual_performances;

CREATE TABLE public.annual_performances
(
    ap_id bigint NOT NULL,
    parcel_id integer,
    plant_id character varying(20) COLLATE pg_catalog."default",
    performance integer,
    CONSTRAINT annual_performances_pkey PRIMARY KEY (ap_id),
    CONSTRAINT parcelid FOREIGN KEY (parcel_id)
        REFERENCES public.parcels ("OID") MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT plantid FOREIGN KEY (plant_id)
        REFERENCES public.plants (name) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;


-- Table: public.parc_cult_plants

-- DROP TABLE public.parc_cult_plants;

CREATE TABLE public.parc_cult_plants
(
    parcel_id integer NOT NULL,
    plant_id character varying(20) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT parc_cult_plants_pkey PRIMARY KEY (parcel_id, plant_id),
    CONSTRAINT parcelid FOREIGN KEY (parcel_id)
        REFERENCES public.parcels ("OID") MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT plantid FOREIGN KEY (plant_id)
        REFERENCES public.plants (name) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;


-- Firm tables for various zone values
SELECT geometry
INTO firm_A
FROM firm
WHERE firm.zone = 'A'

SELECT geometry
INTO firm_AE
FROM firm
WHERE firm.zone = 'AE'

SELECT geometry
INTO firm_X
FROM firm
WHERE firm.zone = 'X'

SELECT geometry
INTO firm_X500
FROM firm
WHERE firm.zone = 'X500'








